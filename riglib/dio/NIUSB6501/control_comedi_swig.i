%module control_comedi_swig
%{
extern unsigned char init(char* dev);
extern int set_bits_in_nidaq(int mask, int data);
%}

extern unsigned char init(char* dev);
extern int set_bits_in_nidaq(int mask, int data);