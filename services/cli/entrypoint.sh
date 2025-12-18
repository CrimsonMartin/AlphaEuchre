#!/bin/bash
set -e

# euchre_core is already installed during Docker build
# Just execute the main command
exec "$@"
