/* Test Code ...
/*    The code provides an example of how to open a serial port and query the
/*    wavelength of an Acton Research Corporation (ARC) Monochromator.
/*
/*    Things to note : 
/*              1. ARC Monochromator's recognize valid commands with a " ok\r".
/*              2. ARC Monochromator's acknowledge in-valid commands with 
/*                 a " ?\r". 
/*              3. Commands sent to the monochromator will not be processed
/*                 until a Carriage Return ( "\r" ) is sent.
/*              4. Never send a LineFeed to the instrument ( "\n" ).
/*              5. This example assumes we are using the first port on a linux 
/*                 system "/dev/ttyS0".
/*              6. Monochromator's echo all characters sent to them.
/*              
/*    Code based on examples provided in :
/*         Serial Programming HOWTO, 
/*              http://www.linuxdoc.org/HOWTO/Serial-Programming-HOWTO/x115.html
/*         Serial Programming Guide for POSIX Operating Systems, 
/*              http://www.easysw.com/~mike/serial/serial.html#2_5
/*    For further reading :
/*         Serial HOWTO,
/*              http://www.linuxdoc.org/HOWTO/Serial-HOWTO.html
/*         Linux I/O port programming mini-HOWTO,
/*              http://www.linuxdoc.org/HOWTO/mini/IO-Port-Programming.html
*/
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>
#include <stdio.h>

/* baudrate settings are defined in <asm/termbits.h>, which is included by <termios.h> */
/* Acton Research Monochromator's
/*       Communicate at 9600 Baud, 8 Bits, No Parity, 1 Stop Bit */
#define BAUDRATE B9600           
/* change this definition for the correct port */
#define MODEMDEVICE "/dev/ttyS0"
#define _POSIX_SOURCE 1 /* POSIX compliant source */         
#define FALSE 0
#define TRUE 1         

volatile int STOP=FALSE;

int main(int argc, char **argv)
{
int fd; /* Pointer to Serial Device */
int res;
struct termios oldtio,newtio;
char buf[255];

/*
  Open modem device for reading and writing and not as controlling tty
  because we don't want to get killed if linenoise sends CTRL-C.
*/
fd = open(MODEMDEVICE, O_RDWR | O_NOCTTY );
if (fd <0) {perror(MODEMDEVICE); exit(-1); }
       
tcgetattr(fd,&oldtio); /* save current serial port settings */
bzero(&newtio, sizeof(newtio)); /* clear struct for new port settings */
       
/*
BAUDRATE: Set bps rate. You could also use cfsetispeed and cfsetospeed.
CRTSCTS : output hardware flow control (only used if the cable has
          all necessary lines. See sect. 7 of Serial-HOWTO)
CS8     : 8n1 (8bit,no parity,1 stopbit)
CLOCAL  : local connection, no modem contol
CREAD   : enable receiving characters
*/
newtio.c_cflag = BAUDRATE | CRTSCTS | CS8 | CLOCAL | CREAD;
     
/*
IGNPAR  : ignore bytes with parity errors
ICRNL   : map CR to NL (otherwise a CR input on the other computer
          will not terminate input)
          otherwise make device raw (no other input processing)
*/
newtio.c_iflag = IGNPAR | ICRNL;
        
/*
Raw output.
*/
newtio.c_lflag = ~(ICANON | ECHO | ECHOE | ISIG);
        
/*
  now clean the modem line and activate the settings for the port
*/
tcflush(fd, TCIFLUSH);
tcsetattr(fd,TCSANOW,&newtio);
       
/* Serial port is now opened and set up */

strcpy(buf,"?nm\r"); /* Send "?nm" command to the Monochromator */
write(fd,buf,4);

/* wait for the monochromator to ackknowledge by sending back an "ok" */
while (STOP==FALSE) {     /* loop until we have a terminating condition */
       /* read blocks program execution until a line terminating character is
          input, even if more than 255 chars are input. If the number
          of characters read is smaller than the number of chars available,
          subsequent reads will return the remaining chars. res will be set
          to the actual number of characters actually read */
    res = read(fd,buf,1);
    buf[res]=0;             /* set end of string, so we can printf */
    if (res > 0)            /* output command results to the console */
       { printf("%s", buf); }
    if (buf[0]=='k') STOP=TRUE;
 }
printf("\n");

/* restore the old port settings */
tcsetattr(fd,TCSANOW,&oldtio);
}
