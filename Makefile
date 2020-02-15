CFLAGS= -g -Wall -Wno-implicit-function-declaration
CC=gcc
LINK    = gcc
INCLUDE =  -I./

CMAIN   =  main.c
CSOURCE =  bfgs.c interp.c matrix.c
CMAINOBJ = $(CMAIN:.c=.o)
COBJS    = $(CSOURCE:.c=.o)

.c.o:
	$(CC)  $(CFLAGS) $(INCLUDE) -c $<

default: cbfgs

cbfgs: $(CMAINOBJ) $(COBJS)
	$(LINK) $(LFLAGS) $(INCLUDE) $(CMAINOBJ) $(COBJS) -o $@

install:
	install -v -m 750 cbfgs ~/bin/cbfgs

clean:
	rm -f *.o *~ a.out cbfgs *.dat *.his

new:
	make clean
	make default
