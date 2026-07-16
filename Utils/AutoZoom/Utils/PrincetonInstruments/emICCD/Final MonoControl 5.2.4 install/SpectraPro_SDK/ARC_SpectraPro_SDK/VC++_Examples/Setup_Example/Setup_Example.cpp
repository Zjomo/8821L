// Setup_Example.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ARC_SpectraPro_dll.h"

// array of instrument enumerations
int Num_Enum;
long Inst_Enum[16];

void Print_FilterInfo(long Filter_Enum)
{
char WorkStr [127];
long MinFilter,MaxFilter,FilterPos;
   
   // display filter controller information  
   if (ARC_Valid_Filter_Enum(Filter_Enum) != 0)
   {
	  printf("Filter Enum : %d \n",Filter_Enum);
	  // Filterwheel Model
      if (ARC_get_Filter_Model_CString(Filter_Enum,*WorkStr) != 0)
	     printf("Filter Model : %s \n",WorkStr); 
	  else
	     printf("Error : Filter Model \n");
	  // Filterwheel Serial number
      if (ARC_get_Filter_Serial_CString(Filter_Enum,*WorkStr) != 0)
	     printf("Filter Serial Number : %s \n",WorkStr); 
	  else
	     printf("Error : Filter Serial Number \n");
	  // Filterwheel Min / Max Filter
      if (ARC_get_Filter_Min_Pos(Filter_Enum,MinFilter) != 0 
	  	   &&
          ARC_get_Filter_Max_Pos(Filter_Enum,MaxFilter) != 0)
	     printf("Max Filter : %d, Min Filter : %d \n",MinFilter,MaxFilter); 
	  else
	     printf("Error : Filter Min/Max \n");
	  // FilterWheel Position
      if (ARC_get_Filter_Position(Filter_Enum,FilterPos) != 0)
	     printf("Filter Position %d \n",FilterPos); 
	  else
	     printf("Error : Filter Position \n");
   }
   else
   {
      printf("Error : Invalid Filter Enum : %d \n",Filter_Enum);
   }
}
void Print_MonoInfo(long Mono_Enum)
{
char WorkStr [127];
double wavelength,HalfAng,DetAng,focallength,minwave,maxwave;
unsigned _int16 DoubleMono;
long Turret,Grating,GratLoop,SlitWidth;
long MinFilter,MaxFilter,FilterPos, GenLoop;

    // validate the monochromator enumeration and print it's information
	if (ARC_Valid_Mono_Enum(Mono_Enum) != 0)
	{
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
		// Mono Double
        if (ARC_get_Mono_Double(Mono_Enum, DoubleMono) != 0)
		{   
			if (DoubleMono != 0)
			   printf("Double Monochromator \n");
		}
		/* Mono Grating information */
		//Grating
        if (ARC_get_Mono_Turret(Mono_Enum, Turret) != 0
			&&
			ARC_get_Mono_Grating(Mono_Enum, Grating) != 0)
		   printf("Mono Turret : %d, Grating %d \n",Turret,Grating);
		else
           printf("Error : Failed to Read Mono Grating/Turret \n");
		// gratings per turret
        if (ARC_get_Mono_Turret_Gratings(Mono_Enum, Grating) != 0)
		   printf("Mono gratings per turret : %d \n",Grating);
		else
           printf("Error : Failed to Read Mono gratings per turret \n");
		// display grating information
		for (GratLoop = 1; GratLoop <= 9; GratLoop++)
		{
		   // if the grating is installed display it's information 	
		   if (ARC_get_Mono_Grating_Installed(Mono_Enum,GratLoop) != 0)
		   { 
              if (ARC_get_Mono_Grating_Blaze_CString(Mono_Enum,GratLoop,*WorkStr) != 0
			      &&
                  ARC_get_Mono_Grating_Density(Mono_Enum,GratLoop,Grating) != 0)
			  {
			     printf("Grating : %d, Densisty : %d, Blaze %s \n",GratLoop,Grating,WorkStr);	
			  }
           }
		}
		/* Mono Diverter Mirror information */
		// Mirrors 1,2 are on the Master, 3,4 are on a slave if present 
		for (GenLoop = 1; GenLoop <= 4; GenLoop++)
		{
           if (ARC_get_Mono_Diverter_Valid(Mono_Enum, GenLoop) != 0)
		      printf("Mono Diverter %d is motorized \n",GenLoop);
           if (ARC_get_Mono_Diverter_Pos_CString(Mono_Enum, GenLoop, *WorkStr) != 0)
		      printf("Mono Divertor %d Position : %s \n",GenLoop,WorkStr);
		}
		/* Mono Motorized Slit information */
		// Slits 1,2,3,4 are on the Master, 5,6,7,8 are on a slave if present
		for (GenLoop = 1; GenLoop <= 8; GenLoop++)
		{
           // slit position
           if (ARC_Mono_Slit_Name_CString(GenLoop,*WorkStr) != 0)
		      printf("Mono Slit %d : %s \n",GenLoop,WorkStr);
		   else
              printf("Error : Failed to Read Mono Slit Name \n");
		   // slit type (motorized, manual, focalplane, ect)
           if (ARC_get_Mono_Slit_Type_CString(Mono_Enum,GenLoop,*WorkStr) != 0)
		      printf("Mono Slit %d Type : %s \n",GenLoop,WorkStr);;
		   // slit width in microns
           if (ARC_get_Mono_Slit_Width(Mono_Enum,GenLoop,SlitWidth) != 0)
		      printf("Mono Slit %d Width : %d microns \n",GenLoop,SlitWidth);
        }
		/* Mono Filter Wheel */
        if (ARC_get_Mono_Filter_Present(Mono_Enum) != 0)
        {
			// Filterwheel Min / Max Filter
            if (ARC_get_Mono_Filter_Min_Pos(Mono_Enum,MinFilter) != 0 
	  	        &&
                ARC_get_Mono_Filter_Max_Pos(Mono_Enum,MaxFilter) != 0)
	           printf("Mono Max Filter : %d, Min Filter : %d \n",MinFilter,MaxFilter); 
	        else
	           printf("Error : Mono Filter Min/Max \n");
	        // FilterWheel Position
            if (ARC_get_Mono_Filter_Position(Mono_Enum,FilterPos) != 0)
	           printf("Mono Filter Position %d \n",FilterPos); 
	        else
	           printf("Error : Mono Filter Position \n");
		}
		/* Mono Embedded Shutter, newwer v6 controllers only */
        if (ARC_get_Mono_Shutter_Valid(Mono_Enum) != 0)
		{
           if (ARC_get_Mono_Shutter_Open(Mono_Enum) != 0)
		      printf("Mono Shutter Open \n");
		   else
              printf("Mono Shutter Closed \n");
		}
		// Mono Min Wavelength nm
		// Mono Max Wavelength nm
		if (ARC_get_Mono_Wavelength_Min_nm(Mono_Enum, minwave) != 0
			&&
			ARC_get_Mono_Wavelength_Cutoff_nm(Mono_Enum, maxwave))
		   printf("Mono Min Wavelength : %f nm, Cutoff Wavelength %f nm \n",minwave,maxwave);
		else
           printf("Error : Failed to Read Mono Min/Cutoff Wavelength \n");
		// Mono Wavelength nm
		if (ARC_get_Mono_Wavelength_nm(Mono_Enum,wavelength) != 0)
			printf("Mono Wavelength : %f nm\n",wavelength);
		else
			printf("Error : Failed to read Mono wavelength in nm\n");
		// Mono Wavelength ang
			if (ARC_get_Mono_Wavelength_ang(Mono_Enum,wavelength) != 0)
			printf("Mono Wavelength : %f ang\n",wavelength);
		else
			printf("Error : Failed to read Mono wavelength in ang\n");
		// Mono Wavelength eV
			if (ARC_get_Mono_Wavelength_eV(Mono_Enum,wavelength) != 0)
			printf("Mono Wavelength : %f eV\n",wavelength);
		else
			printf("Error : Failed to read Mono wavelength in eV \n");
		// Mono Wavelength micron
			if (ARC_get_Mono_Wavelength_micron(Mono_Enum,wavelength) != 0)
			printf("Mono Wavelength : %f microns\n",wavelength);
		else
			printf("Error : Failed to read Mono wavelength in microns \n");
	}
	else
	{
		printf("Error : Invalid Mono Enum : %d \n",Mono_Enum);
	}
}

void Setup_Filter (long Filter_Enum)
{
long NewFilter_Enum;
	// open the Filter Controller
	printf("Opening Filter Controller, Enum : %d \n",Filter_Enum);

	if (ARC_Open_Filter(Filter_Enum,NewFilter_Enum) != 0) 
	{
	   // we opened the device
	   Print_FilterInfo(NewFilter_Enum);
	   // store the Mono enumeration data so we can use it later
	   Inst_Enum[Num_Enum] = NewFilter_Enum;
	   Num_Enum++;	
	}
	else
	{
	   printf("Error : Failed to open Filter Controller at Enum : %d \n",Filter_Enum);
	}
}
void Setup_Mono (long Mono_Enum)
{
long NewMono_Enum;
	// open the Monochromator / Spectrograph
	printf("Opening Monochromator / SpectraGraph, Enum : %d \n",Mono_Enum);

	if (ARC_Open_Mono(Mono_Enum,NewMono_Enum) != 0) 
	{
	   // we opened the device
	   Print_MonoInfo(NewMono_Enum);
	   // store the Mono enumeration data so we can use it later
	   Inst_Enum[Num_Enum] = NewMono_Enum;
	   Num_Enum++;	
	}
	else
	{
	   printf("Error : Failed to open Mono at Enum : %d \n",Mono_Enum);
	}
}
void Setup_Instruments ()
{
long Num_Found;
long Cur_Enum;
char WorkStr [127]; 

   // no instruments have bee setup
   Num_Enum = 0;
   printf("Searching for Acton Research Corp. Monochromator / Filter Controllers \n");
   if (ARC_Search_For_Mono(Num_Found) != 0)
   {
      // display how many devices located
      printf("%d Acton Research Corp. Devices Found! \n",Num_Found);
	  
	  // now open and setup all the instruments
	  // the Acton enum start with 0, if 3 devices are found it means
	  // Enums 0,1,2 are the valid device numbers
	  for (Cur_Enum = 0; Cur_Enum < Num_Found; Cur_Enum++)
      {
		 {
            // check for an Acton Monochromator / SpectroGraph
		    if (ARC_get_Mono_preOpen_Model_CString(Cur_Enum,*WorkStr) != 0)
			{
				Setup_Mono(Cur_Enum);
			}
			else
			{
               // check for an Acton Filter Controller
			   if (ARC_get_Filter_preOpen_Model_CString(Cur_Enum,*WorkStr) != 0)
			   {
				   Setup_Filter (Cur_Enum);
               }
			   else
			   {
				   printf("Error : Enum %d Invalid \n",Cur_Enum);
			   }
			}
		 }
	  }
   }
   else
   {
      printf("No Acton Research Corp. devices located \n");
   }
}
void Close_Instruments ()
{
long Cur_Enum;

   printf("Closing Open Devices\n");
   if (Num_Enum > 0)
   {
      for (Cur_Enum = 0; Cur_Enum < Num_Enum; Cur_Enum++)
      {
         printf("Closing Enum : %d \n",Inst_Enum[Cur_Enum]);
		 ARC_Close_Mono(Inst_Enum[Cur_Enum]);
	  }
   }
   else
   {
	   printf("Error : No devices to close \n");
   }
}

int main(int argc, char* argv[])
{
long MajorVal,MinorVal,BuildVal; ///return values from ARC_Ver

  // dynamically load the dll 
  if (Setup_ARC_SpectraPro_dll() == TRUE)
  {
     printf("DLL Loaded!\n");
	 // obtain and the version of the dll being used
     if (ARC_Ver(MajorVal,MinorVal,BuildVal) != 0)
	 {
        printf("ARC_SpectraPro.dll Ver : %d.%d.%d \n", MajorVal, MinorVal, BuildVal);
		// locate and setup all Acton Instruments
		Setup_Instruments ();
		// we are done with the instruments, close what we have opened
		Close_Instruments ();
	 }
     else
     {
        printf("Error : ARC_Ver Failed!\n");
	 }
     // completed talking to Acton Instruments, unload the DLL
     UnLoad_ARC_SpectraPro_dll();
     }
  else
  {
     printf("Error : DLL not Loaded!\n");
  }
  return 0;
}


