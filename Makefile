# TODO: Try -ffast-math
CC=cc
CFLAGS=-std=c99 -Wall -Wextra -Werror -Wno-unused-parameter
EXTRA_CFLAGS?=-O1
LDFLAGS=

TARGETS=demo headless
SOURCES=$(shell echo *.c)
COMMON_OBJECTS=timing.o solver.o

.PHONY: clean
all: $(TARGETS)

demo: demo.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $^ -o $@ $(LDFLAGS) -lGL -lGLU -lglut

headless: headless.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $^ -o $@ $(LDFLAGS)

asm: solver.o
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) \
	-fno-asynchronous-unwind-tables -fno-exceptions -fverbose-asm -S solver.c

%.o: %.c %.h
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) -c $< -o $@
headless.o: headless.c # TODO: Generalize %.o to include headless
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) -c $< -o $@

runperf: headless
	sudo perf stat -e cache-references,cache-misses,L1-dcache-load-misses,L1-dcache-loads,L1-dcache-stores,L1-icache-load-misses,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses ./headless

.PHONY: clean
clean:
	rm -f $(TARGETS) *.o .depend solver.s *~

.depend: *.[ch]
	$(CC) -MM $(SOURCES) >.depend

-include .depend
