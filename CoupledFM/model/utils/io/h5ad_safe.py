"""H5AD 写入安全化工具。

pandas >= 2.x 在一些环境里会把 ``AnnData.obs`` 的字符串列 / 索引装成 ``string[pyarrow]``
或直接装成 ``ArrowStringArray``，而 ``anndata`` 的 h5ad writer 目前还没注册这些
extension dtype → ``IORegistryError: No method registered for writing
pandas.arrays.ArrowStringArray into <class 'h5py._hl.group.Group'>``。

本模块提供两个帮手：

* :func:`sanitize_for_h5ad` —— 就地把 ``adata.obs/var`` 的索引与字符串/扩展 dtype 列
  物化为普通 numpy ``object``；bool/数值/``pd.Categorical`` 保持原样。
* :func:`safe_write_h5ad` —— 先 sanitize 再调用 ``adata.write_h5ad``，兜底重试。

所有涉及 ``write_h5ad`` 的脚本（GT 编码、切片保存等）都应统一调用这里。
下游 ``LazyH5AnnData`` / ``read_obs_meta`` 直接用 h5py 读，不受影响。
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


def _to_object_index(idx: pd.Index) -> pd.Index:
    """索引物化为 numpy object 数组，不留 Arrow/string dtype 残留。

    注意：pandas 3.x 默认 ``future.infer_string=True``，``pd.Index(object_array)`` 会
    **自动转回 ArrowStringArray**。必须显式 ``dtype=object`` 才能强制保持 numpy object。
    """
    try:
        values = [str(x) for x in idx]
    except Exception:
        values = [str(x) for x in idx.tolist()]
    return pd.Index(np.asarray(values, dtype=object), name=idx.name, dtype=object)


def _object_series_like(ser: pd.Series) -> pd.Series:
    """把字符串/Arrow extension Series 物化为 object dtype Series。"""
    try:
        values = np.asarray(ser.tolist(), dtype=object)
    except Exception:
        values = np.asarray([str(x) if x is not None else None for x in ser], dtype=object)
    return pd.Series(values, index=ser.index.copy(), dtype=object, name=ser.name)


def _needs_object(ser: pd.Series) -> bool:
    """判断一列是否需要转成 object 才能写入 h5ad。"""
    dtype = ser.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return False
    if pd.api.types.is_numeric_dtype(ser) or pd.api.types.is_bool_dtype(ser):
        return False
    if pd.api.types.is_datetime64_any_dtype(ser) or pd.api.types.is_timedelta64_dtype(ser):
        return False
    if pd.api.types.is_string_dtype(ser):
        return True
    if str(dtype).startswith("string"):
        return True
    if getattr(dtype, "storage", None) == "pyarrow":
        return True
    try:
        arr_name = type(ser.array).__name__
    except Exception:
        arr_name = ""
    if "Arrow" in arr_name:
        return True
    if getattr(dtype, "na_value", None) is pd.NA:
        return True
    return False


def _materialize_df(df: pd.DataFrame) -> pd.DataFrame:
    """返回一个全字符串/object 物化的新 DataFrame，索引也用 object 物化。

    构造 DataFrame 时需再次强制每个字符串列用 ``dtype=object``，否则 pandas 3.x 下
    ``future.infer_string=True`` 会把 object 数组重新包装成 Arrow。
    """
    if df is None:
        return df
    new_idx = _to_object_index(df.index)
    cols = {}
    for col in df.columns:
        ser = df[col]
        if isinstance(ser.dtype, pd.CategoricalDtype):
            cats = ser.cat.categories
            if _needs_object(pd.Series(cats)):
                new_cats = pd.Index(
                    np.asarray([str(x) for x in cats], dtype=object),
                    dtype=object,
                )
                ser = ser.cat.rename_categories(dict(zip(cats, new_cats)))
            cols[col] = ser.values
            continue
        if pd.api.types.is_numeric_dtype(ser) or pd.api.types.is_bool_dtype(ser):
            cols[col] = np.asarray(ser.values)
            continue
        if pd.api.types.is_datetime64_any_dtype(ser) or pd.api.types.is_timedelta64_dtype(ser):
            cols[col] = np.asarray(ser.values)
            continue
        if _needs_object(ser):
            arr = np.asarray(
                [
                    str(x)
                    if x is not None and not (isinstance(x, float) and pd.isna(x))
                    else None
                    for x in ser
                ],
                dtype=object,
            )
            cols[col] = pd.Series(arr, index=df.index.copy(), dtype=object, name=col)
        else:
            cols[col] = np.asarray(ser.values)
    out = pd.DataFrame(index=new_idx)
    out.index.name = df.index.name
    for col, values in cols.items():
        if isinstance(values, pd.Series):
            # 已经是 dtype=object Series
            out[col] = values.values
            # 再次强制 object，防止赋值时被 pandas infer 回 Arrow
            if not pd.api.types.is_object_dtype(out[col]):
                out[col] = pd.Series(values.values, index=out.index.copy(), dtype=object)
        else:
            out[col] = values
    return out


def sanitize_for_h5ad(adata) -> None:
    """就地修复 AnnData 使其能被 anndata 的 h5ad writer 序列化。

    - ``obs.index`` / ``var.index`` 物化为 object numpy 数组；
    - ``obs``/``var`` 的字符串或 pandas extension 列转为 object；
    - ``uns`` 里 0-d numpy 数组 / pandas Timestamp 转为 Python 标量（常见 Arrow 副作用）。
    """
    if adata is None:
        return

    try:
        if adata.obs is not None:
            adata.obs = _materialize_df(adata.obs)
    except Exception:
        pass
    try:
        if adata.var is not None:
            adata.var = _materialize_df(adata.var)
    except Exception:
        pass

    uns = getattr(adata, "uns", None)
    if isinstance(uns, dict):
        for k, v in list(uns.items()):
            try:
                if isinstance(v, np.ndarray) and v.ndim == 0:
                    uns[k] = v.item()
                elif isinstance(v, pd.Timestamp):
                    uns[k] = v.to_pydatetime().isoformat()
            except Exception:
                pass


def safe_write_h5ad(adata, path: Union[str, Path], compression: str = "gzip", **kwargs) -> None:
    """sanitize + write_h5ad；失败时在 ``mode.string_storage='python'`` 上下文里再重试一次。"""
    sanitize_for_h5ad(adata)
    try:
        adata.write_h5ad(str(path), compression=compression, **kwargs)
        return
    except Exception:
        # 兜底：在 string_storage='python' 环境下重新物化 DataFrame 再写。
        # pandas 3.x 下 future.infer_string=True 会把 object 重新包装；此处临时关掉。
        prev_storage = pd.get_option("mode.string_storage")
        prev_infer = None
        try:
            prev_infer = pd.get_option("future.infer_string")
        except Exception:
            pass
        try:
            pd.set_option("mode.string_storage", "python")
            if prev_infer is not None:
                pd.set_option("future.infer_string", False)
            sanitize_for_h5ad(adata)
            adata.write_h5ad(str(path), compression=compression, **kwargs)
        finally:
            pd.set_option("mode.string_storage", prev_storage)
            if prev_infer is not None:
                pd.set_option("future.infer_string", prev_infer)


__all__ = ["sanitize_for_h5ad", "safe_write_h5ad"]
