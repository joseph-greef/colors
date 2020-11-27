#!/bin/bash

STEPS=$1
COLORS=$2
echo $STEPS
echo $COLOs
export QUERYSTR="colors=$COLORS&steps=$STEPS"
php -e -r 'parse_str($_SERVER["QUERYSTR"], $_GET); include "gradient.php";'
