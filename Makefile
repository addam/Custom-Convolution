#  Use gimptool-2.0 to sett these variables
GIMPTOOL = gimptool-2.0
PLUGIN_INSTALL = $(GIMPTOOL) --install-bin
GCC = gcc
LIBS = $(shell pkg-config fftw3f gimp-2.0 gimpui-2.0 gtk+-2.0 --libs) -lm
CFLAGS = -std=c99 -O2 $(shell pkg-config fftw3f gimp-2.0 gimpui-2.0 gtk+-2.0 --cflags)

all: convolution

convolution: plugin.c
	$(GCC) $(CFLAGS) -o $@ $^ $(LIBS)

install: convolution
	$(PLUGIN_INSTALL) $^
	 
clean:
	rm -f gimp-test
