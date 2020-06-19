# TODO: Dummy CUDA target, integrate properly into the Makefile
# See https://devblogs.nvidia.com/separate-compilation-linking-cuda-device-code/
# for more compiling and linking ideas, see -dc and -dlink to mix gcc and nvcc

# Toggle comments for switching headless/demo compilation
cudademo: clean
	nvcc -DCUDA -rdc=true -arch=sm_61 -L /usr/lib/x86_64-linux-gnu -x cu demo.c timing.c solver.c solver_cuda.c -o demo -lGL -lGLU -lglut --compiler-options=-fopenmp
cudaheadless: clean
	nvcc -DCUDA -rdc=true -arch=sm_61 -L /usr/lib/x86_64-linux-gnu -x cu headless.c timing.c solver.c solver_cuda.c -o headless --compiler-options=-fopenmp

BUILD=intrinsics # nonvect | intrinsics | ispc

fastest_cflags:=-Ofast -march=native -floop-nest-optimize -funroll-loops -flto -g
cflags.nonvect:=$(fastest_cflags)
cflags.intrinsics:=$(fastest_cflags) -DINTRINSICS
cflags.ispc:=$(fastest_cflags) -DISPC

build_object.nonvect:=solver_nonvect.o
build_object.intrinsics:=solver_intrinsics.o
build_object.ispc:=solver_ispc.o

CC=cc
ISPC=ispc
ISPCFLAGS=--target=avx2-i32x8
override CFLAGS:=-std=c99 -Wall -Wextra -Wshadow -Wno-unused-parameter $(cflags.$(BUILD)) $(CFLAGS)
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
