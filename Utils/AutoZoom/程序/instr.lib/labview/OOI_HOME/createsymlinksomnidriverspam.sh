#!/bin/bash


if [ -h /usr/lib/libOmniDriver.so ]; then
   rm -f /usr/lib/libOmniDriver.so
fi
ln -s $1/OOI_HOME/libOmniDriver.so /usr/lib/libOmniDriver.so

if [ -h /usr/lib/libSPAM.so ]; then
   rm -f /usr/lib/libSPAM.so
fi
ln -s $1/OOI_HOME/libSPAM.so /usr/lib/libSPAM.so

if [ -h /usr/lib/libcommon.so ]; then
   rm -f /usr/lib/libcommon.so
fi
ln -s $1/OOI_HOME/libcommon.so /usr/lib/libcommon.so

if [ -h /usr/include/omnidriver ]; then
   rm -f /usr/include/omnidriver
fi
ln -s $1/include /usr/include/omnidriver
