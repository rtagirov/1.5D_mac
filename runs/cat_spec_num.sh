#!/bin/bash

cd $1

for f in *; do

    if [ $(wc -l $f | awk '{print $1}') = 120 ]; then

        echo $f

#        awk '{print $2}' $f >> $1.spec
        echo $f >> ../$1.spec
        cat  $f >> ../$1.spec

    fi

done

cd ..
