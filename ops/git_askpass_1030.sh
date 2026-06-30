#!/usr/bin/env bash
case "$1" in
  *Username*) printf '%s\n' "x-access-token" ;;
  *Password*) printf '%s\n' "${GHTOKEN_1030:?GHTOKEN_1030 is not set}" ;;
  *) printf '\n' ;;
esac
