#  Use gimptool-2.0 to sett these variables
GIMPTOOL=gimptool-2.0
PLUGIN_INSTALL=$(GIMPTOOL) --install-bin
GCC=g++
LIBS=$(shell pkg-config fftw3 gimp-2.0 gimpui-2.0 gtk+-2.0 --libs)
CFLAGS=-O2 -g $(shell pkg-config fftw3 gimp-2.0 gimpui-2.0 gtk+-2.0 --cflags)

all: gimp-test

# Use of pkg-config is the recommended way
gimp-test: plugin.c
	$(GCC) $(CFLAGS) $(LIBS) -o gimp-test plugin.c  

# To avoid gimptool use, just copy the fourier in the directory you want
install: gimp-test
	$(PLUGIN_INSTALL) gimp-test
	 
clean:
	rm -f gimp-test
