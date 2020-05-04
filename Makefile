# Example run make headless BUILD=fast
# TODO: For some reason BUILD initialization needs to be done explicitly in args
BUILD=fast
cflags.common:=
cflags.fast:=-O3 -xHost -fp-model fast=2 -no-prec-div

CC=icc
override CFLAGS:=-std=c99 -Wall -Wextra -Werror -Wshadow -Wno-unused-parameter $(cflags.$(BUILD)) $(CFLAGS)
LDFLAGS=

TARGETS=demo headless
SOURCES=$(shell echo *.c)
COMMON_OBJECTS=timing.o solver.o

.PHONY: clean
all: $(TARGETS)
	source /opt/ipsxe/2019u1/bin/compilervars.sh intel64

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

.depend: *.[ch]
	$(CC) -MM $(SOURCES) > .depend

-include .depend

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(TARGETS) *.o .depend solver.s *~
