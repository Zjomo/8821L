// Move_Grating.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ARC_SpectraPro_dll.h"

// array of instrument enumerations
int Num_Enum;
long Inst_Enum[16];


void Print_MonoInfo(long Mono_Enum)
{
	char WorkStr [127];
	double wavelength,HalfAng,DetAng,focallength,minwave,maxwave;
	unsigned _int16 DoubleMono;
	long Turret,Grating,GratLoop,SlitWidth;
	long MinFilter,MaxFilter,FilterPos, GenLoop;

    // validate the monochromator enumeration and print it's information
	if (ARC_Valid_Mono_Enum(Mono_Enum) != 0) 	{
		printf("Mono Enum : %d \n",Mono_Enum);
		// Mono Model
        if (ARC_get_Mono_Model_CString(Mono_Enum, *WorkStr) != 0)
		   printf("Mono Model : %s \n",WorkStr);
		else
           printf("Error : Failed to Read Mono Model \n");
		// Mono Serial Number
		if (ARC_get_Mono_Serial_CString(Mono_Enum, *WorkStr) != 0)
			printf("Mono Serial Number : %s \n",WorkStr);
		else
			printf("Error : Failed to Read Mono Serial Number \n");
		/* Mono Optical Information */
		// Focallength
		if (ARC_get_Mono_Focallength(Mono_Enum, focallength) != 0)
		   printf("Mono focallength : %f mm\n",focallength);
		else
           printf("Error : Failed to Read Mono Focallength \n");
        // Half Angle
		if (ARC_get_Mono_HalfAngle(Mono_Enum, HalfAng) != 0)
		   printf("Mono Half Angle : %f radians\n",HalfAng);
		else
           printf("Error : Failed to Read Mono Half Angle \n");
        // Detector Angle
		if (ARC_get_Mono_DetectorAngle(Mono_Enum, DetAng) != 0)
			printf("Mono Detector Angle : %f radians\n",DetAng);
		else
           printf("Error : Failed to Read Mono Detector Angle \n");

		//Grating
        if (ARC_get_Mono_Turret(Mono_Enum, Turret) != 0
			&& 	ARC_get_Mono_Grating(Mono_Enum, Grating) != 0)
		   printf("Mono Turret : %d, Grating %d \n",Turret,Grating);
		else
           printf("Error : Failed to Read Mono Grating/Turret \n");
		// gratings per turret
        if (ARC_get_Mono_Turret_Gratings(Mono_Enum, Grating) != 0)
		   printf("Mono gratings per turret : %d \n",Grating);
		else
           printf("Error : Failed to Read Mono gratings per turret \n");
		// display grating information

		for (GratLoop = 1; GratLoop <= 9; GratLoop++) 		{
		   // if the grating is installed display it's information 	
		   if (ARC_get_Mono_Grating_Installed(Mono_Enum,GratLoop) != 0)   { 
              if (ARC_get_Mono_Grating_Blaze_CString(Mono_Enum,GratLoop,*WorkStr) != 0
			      &&  ARC_get_Mono_Grating_Density(Mono_Enum,GratLoop,Grating) != 0)  {
			     printf("Grating : %d, Densisty : %d, Blaze %s \n",GratLoop,Grating,WorkStr);	
			  }
           }
		}
	} else {
		printf("Error : Invalid Mono Enum : %d \n",Mono_Enum);
	}
}

void Setup_Mono (long Mono_Enum)
{
	long NewMono_Enum;
	// open the Monochromator / Spectrograph
	printf("Opening Monochromator / SpectraGraph, Enum : %d \n",Mono_Enum);

	if (ARC_Open_Mono(Mono_Enum,NewMono_Enum) != 0) {    // we opened the device
	   Print_MonoInfo(NewMono_Enum); 	   // store the Mono enumeration data so we can use it later
	   Inst_Enum[Num_Enum] = NewMono_Enum;
	   Num_Enum++;	
	} else 	{
	   printf("Error : Failed to open Mono at Enum : %d \n",Mono_Enum);
	}
}

void Setup_Instruments ()
{
	long Num_Found;
	long Cur_Enum;

	char WorkStr [127]; 
	int GratLoop, tempLoop;
   // no instruments have bee setup
   Num_Enum = 0;
   printf("Searching for Acton Research Corp. Monochromator / Filter Controllers \n");
   if (ARC_Search_For_Mono(Num_Found) != 0)    {
      // display how many devices located
      printf("%d Acton Research Corp. Devices Found! \n",Num_Found);
	  
	  // now open and setup all the instruments
	  // the Acton enum start with 0, if 3 devices are found it means
	  // Enums 0,1,2 are the valid device numbers
	  for (Cur_Enum = 0; Cur_Enum < Num_Found; Cur_Enum++)   {
        // check for an Acton Monochromator / SpectroGraph
	    if (ARC_get_Mono_preOpen_Model_CString(Cur_Enum,*WorkStr) != 0)	{
			Setup_Mono(Cur_Enum);
			// We've set up that mono, just move it's grating.
			for (GratLoop = 1; GratLoop <= 3; GratLoop++) 		{
			   if (ARC_get_Mono_Grating_Installed(Cur_Enum,GratLoop) != 0)   { 
				   tempLoop = (GratLoop == 3 ? 1 : GratLoop+1);
				   if (ARC_get_Mono_Grating_Installed(Cur_Enum,tempLoop) != 0)   { 
						// go to first, then go to next.			   
						printf("Going to Grating %d\n", GratLoop);
						if (ARC_set_Mono_Grating(Cur_Enum,(long)GratLoop)) {
							printf("Going to Grating %d\n",tempLoop);
							ARC_set_Mono_Grating(Cur_Enum,(long)tempLoop);
						}
				   }
			   }
			}


		}
	  }
   } else {
      printf("No Acton Research Corp. devices located \n");
   }
}
void Close_Instruments ()
{
	long Cur_Enum;

   printf("Closing Open Devices\n");
   if (Num_Enum > 0)    {
      for (Cur_Enum = 0; Cur_Enum < Num_Enum; Cur_Enum++)      {
         printf("Closing Enum : %d \n",Inst_Enum[Cur_Enum]);
		 ARC_Close_Mono(Inst_Enum[Cur_Enum]);
	  }
   } else {
	   printf("Error : No devices to close \n");
   }
}

int main(int argc, char* argv[])
{
	long MajorVal,MinorVal,BuildVal; ///return values from ARC_Ver

	// dynamically load the dll 
	if (Setup_ARC_SpectraPro_dll() == TRUE)  {
		printf("DLL Loaded!\n");
		// obtain and the version of the dll being used
		if (ARC_Ver(MajorVal,MinorVal,BuildVal) != 0) {
			printf("ARC_SpectraPro.dll Ver : %d.%d.%d \n", MajorVal, MinorVal, BuildVal);
			// locate and setup all Acton Instruments
			Setup_Instruments ();
			// we are done with the instruments, close what we have opened
			Close_Instruments ();
		 }  else   {
			printf("Error : ARC_Ver Failed!\n");
		 }
		 // completed talking to Acton Instruments, unload the DLL
		 UnLoad_ARC_SpectraPro_dll();
    }  else  {
		printf("Error : DLL not Loaded!\n");
	}
	return 0;
}


