
DEPS = util_opencl.o

# pkg-config --cflags gegl-wrapper
CFLAGS=-pthread -I/home/qmoto/dev/GEGL_build/include/gegl-wrapper \
	-Ivendor/include \
	-I/home/qmoto/dev/GEGL_build/include/gegl-0.3 \
	-I/home/qmoto/dev/GEGL_build/include/babl-0.1 \
	-I/usr/include/json-glib-1.0 -I/usr/include/gio-unix-2.0/ \
	-I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include -I/usr/lib64/glib-2.0/include
# pkg-config --libs gegl-wrapper
LDFLAGS=-L/home/qmoto/dev/GEGL_build/lib -lgegl-wrapper \
	-lgegl-0.3 -lgegl-npd-0.3 -lm -Wl,--export-dynamic \
	-lgmodule-2.0 -pthread -ljson-glib-1.0 -lgio-2.0 -lgobject-2.0 \
	-lglib-2.0 -lbabl-0.1
LDOPENCLFLAGS=-Lvendor/lib64 -Lvendor/lib/x86_64 -lOpenCL

all: GEGL_invertgamma

GEGL_invertgamma: GEGL_invertgamma.o $(DEPS)
	g++ $< $(DEPS) $(LDFLAGS) $(LDOPENCLFLAGS) -o GEGL_invertgamma

%.o : %.c
	g++ -c -g $(CFLAGS) $< -o $@
%.o : %.cpp
	g++ -c -g $(CFLAGS) $< -o $@

check: clean GEGL_invertgamma
	./GEGL_invertgamma Y7zRVPd.jpg test.jpg

clean:
	rm -f *.o GEGL_invertgamma
	rm -f test.jpg

GEGL_invertgamma.o : GEGL_invertgamma.c
util_opencl.o : util_opencl.cpp

.PHONY: clean check
