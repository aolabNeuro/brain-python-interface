#include <stdio.h>	/* for printf() */
#include <comedilib.h>
#include <unistd.h>

//this c program needs to be compiled first
//cc control_comedi.c -lcomedi -lm -o control_comedi

/*
To-do
1. error checking. 
*/

int subdev = 0;		/* change this to your input subdevice */
int range = 0;		/* more on this later */
int aref = AREF_GROUND;	/* more on this later */

int main(int argc,char *argv[])
{

	//then deal with the comedi stuff
	comedi_t *it;
	int retval;
	it = comedi_open("/dev/comedi0");
	if(it == NULL) {
		comedi_perror("comedi_open");
		return 1;
	}

	//let's get the mode
	int mode;
	sscanf(argv[1], "%d",&mode);

	int set_bit;
	int chan;
	int mask;
	int data;

	switch(mode){
		case 0: //write to channel
		    //process the arguments, eh. 
			//argv[0] is the program name, not useful


			sscanf(argv[2], "%d", &chan);
			sscanf(argv[3],"%d", &set_bit);
			printf("channel set to %d, bit set to %d \n", chan,set_bit);
			retval = comedi_dio_write(it, subdev, chan, set_bit);
			break;
			
		case 1: //write to data mask
			//


			sscanf(argv[2], "%x", &mask);
			sscanf(argv[3],"%x", &data);
			retval = comedi_dio_bitfield(it, subdev, mask, &data);
			printf("channel set to %x, data set to %x \n", mask,  data);

			break;


	}

	printf("status from the action to %d\n", retval);


		
	if(retval < 0) {
		comedi_perror("comedi_dio_write");
		return 1;
	}
	
	printf("set channel 0 to %d\n", set_bit);	


	return 0;
}
