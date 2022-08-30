#!/bin/bash
EXE	= p3
CC	= g++
SRC	= ./src
BIN	= ./bin

all: $(BIN)/$(EXE)
 
$(BIN)/$(EXE): 
	$(CC) -O3 -o $(BIN)/$(EXE) $(SRC)/main.cpp

