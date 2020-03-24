CC=cc
CFLAGS=-std=c99 -Wall -Wextra -Werror -Wno-unused-parameter
EXTRA_CFLAGS?=-O1
LDFLAGS=

TARGETS=demo headless
SOURCES=$(shell echo *.c)
COMMON_OBJECTS=timing.o solver.o

all: $(TARGETS)

demo: demo.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $^ -o $@ $(LDFLAGS) -lGL -lGLU -lglut

headless: headless.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) $^ -o $@ $(LDFLAGS)

asm: solver.o
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) \
	-fno-asynchronous-unwind-tables -fno-exceptions -fverbose-asm -S solver.c

clean:
	rm -f $(TARGETS) *.o .depend solver.s *~

.depend: *.[ch]
	$(CC) -MM $(SOURCES) >.depend

-include .depend

.PHONY: clean all
