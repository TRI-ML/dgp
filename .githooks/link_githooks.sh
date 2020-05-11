#!/bin/bash
# Copyright 2019 Toyota Research Institute.  All rights reserved.
chmod a+x .githooks/pre-commit
ln -s -f ../../.githooks/pre-commit .git/hooks/pre-commit
chmod a+x .githooks/pre-push
ln -s -f ../../.githooks/pre-push .git/hooks/pre-push
