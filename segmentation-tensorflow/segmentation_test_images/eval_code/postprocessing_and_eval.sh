#!/bin/bash

matlab -nodisplay -singleCompThread -r "postprocessing_and_eval(); exit;" </dev/null >log.txt

exit 0
