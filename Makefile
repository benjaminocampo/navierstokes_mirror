# Example run make headless BUILD=fast
# TODO: For some reason BUILD initialization needs to be done explicitly in args
BUILD=fast
cflags.common:=
cflags.fast:=-Ofast -march=native -floop-nest-optimize -funroll-loops -flto -g

CC=cc
override CFLAGS:=-std=c99 -Wall -Wextra -Werror -Wshadow -Wno-unused-parameter $(cflags.$(BUILD)) $(CFLAGS)
ISPC=ispc
ISPCFLAGS=--target=sse4-i32x8
LDFLAGS=

TARGETS=demo headless
SOURCES=$(shell echo *.c)
COMMON_OBJECTS=timing.o solver.o solver_ispc.o

all: $(TARGETS)

demo: demo.o $(COMMON_OBJECTS)
        $(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lGL -lGLU -lglut

headless: headless.o $(COMMON_OBJECTS)
        $(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

asm: solver.o
        $(CC) $(CFLAGS) -fno-asynchronous-unwind-tables -fno-exceptions -S solver.c

runperf: headless
        sudo perf stat \
        -e \
        cache-references,\
        cache-misses,\
        L1-dcache-stores,\
        L1-dcache-store-misses,\
        LLC-stores,\
        LLC-store-misses,\
        page-faults,\
        cycles,\
        instructions,\
        branches,\
        branch-misses \
        -ddd \
        ./headless

%_ispc.h: %.ispc
        $(ISPC) $< -h $@ $(ISPCFLAGS)

%_ispc.o: %.ispc
        $(ISPC) $< -o $@ $(ISPCFLAGS)

.depend: *.[ch]
        $(CC) -MM $(SOURCES) > .depend

-include .depend

%.o: %.c
        $(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

%.s: %.c
        $(CC) $(CFLAGS) $^ -S -masm=intel

.PHONY: clean ispc_server
clean:
        rm -f $(TARGETS) *.o *.asm .depend solver.s *~

ispc_server: ISPC='/opt/ispc/1.12.0/bin/ispc'
ispc_server: ISPCFLAGS='--target=avx2-i32x8'
ispc_server: CFLAGS:=$(CFLAGS) -fPIC -no-pie
ispc_server: headless