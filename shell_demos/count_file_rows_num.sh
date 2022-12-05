#!/bin/sh
echo "this is my first shell"
cat='gungun'
_cat='gungun'
oneCat='gun'
echo ${cat}
readonly _cat
echo $_cat
unset cat
echo ${cat}

