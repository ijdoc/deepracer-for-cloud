#!/usr/bin/env bash

error () {
  echo -e "\n${WARN_COLOR}[\xE2\x9C\x98] $1\n${NC}" >&2
  exit 1
}

test_command_outcome () {
  if test $? -eq 0
  then
    echo -e "${OK_COLOR}[\xE2\x9C\x94] $1${NC}"
  else
    error "$1 failed. Correct the issues and try again."
  fi
}

# Colors
WARN_COLOR='\033[0;33m'
OK_COLOR='\033[0;92m'
NC='\033[0m' # No Color