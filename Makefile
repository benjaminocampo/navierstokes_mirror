BUILD=intrinsics

fastest_cflags:=-Ofast -march=native -floop-nest-optimize -funroll-loops -flto -g
cflags.nonvect:=$(fastest_cflags)
cflags.intrinsics:=$(fastest_cflags) -DINTRINSICS
cflags.ispc:=$(fastest_cflags) -DISPC

build_object.nonvect:=solver_nonvect.o # TODO: Make this work by bringing in the sequential code
build_object.intrinsics:=solver_intrinsics.o
build_object.ispc:=solver_ispc.o

CC=cc
ISPC=ispc
ISPCFLAGS=--target=avx2-i32x8
override CFLAGS:=-std=c99 -Wall -Wextra -Werror -Wshadow -Wno-unused-parameter $(cflags.$(BUILD)) $(CFLAGS)
LDFLAGS=

TARGETS=demo headless
SOURCES=$(shell echo *.c)
COMMON_OBJECTS=timing.o solver.o $(build_object.$(BUILD))


all: $(TARGETS)

demo: demo.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lGL -lGLU -lglut

headless: headless.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

asm: solver.o
	$(CC) $(CFLAGS) -fno-asynchronous-unwind-tables -fno-exceptions -S solver.c

%_ispc.h: %.ispc
	$(ISPC) $< -h $@ $(ISPCFLAGS)

%_ispc.o: %.ispc
	$(ISPC) $< -o $@ $(ISPCFLAGS)

.depend: *.[ch]
	$(CC) $(CFLAGS) -MM $(SOURCES) > .depend

-include .depend

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

%.s: %.c
	$(CC) $(CFLAGS) $^ -S -masm=intel

.PHONY: clean
clean:
	rm -f $(TARGETS) *.o *.asm .depend solver.s *~

# TODO: This should probably not exist
.PHONY: ispc_server
ispc_server: ISPC='/opt/ispc/1.12.0/bin/ispc'
ispc_server: ISPCFLAGS='--target=avx2-i32x8'
ispc_server: CFLAGS:=$(CFLAGS) -fPIC -no-pie
ispc_server: headless
