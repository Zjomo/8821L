// Setup_Example.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ARC_FilterWheel_dll.h"

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
void Setup_Instruments ()
{
long Num_Found;
long Cur_Enum;
char WorkStr [127]; 

   // no instruments have bee setup
   Num_Enum = 0;
   printf("Searching for Acton Research Corp. Filter Controllers \n");
   if (ARC_Search_For_Filter(Num_Found) != 0)
   {
      // display how many devices located
      printf("%d Acton Research Corp. Devices Found! \n",Num_Found);
	  
	  // now open and setup all the instruments
	  // the Acton enum start with 0, if 3 devices are found it means
	  // Enums 0,1,2 are the valid device numbers
	  for (Cur_Enum = 0; Cur_Enum < Num_Found; Cur_Enum++)
      {
		 {
            
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
		 ARC_Close_Filter(Inst_Enum[Cur_Enum]);
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
  if (Setup_ARC_FilterWheel_dll() == TRUE)
  {
     printf("DLL Loaded!\n");
	 // obtain and the version of the dll being used
     if (ARC_Ver(MajorVal,MinorVal,BuildVal) != 0)
	 {
        printf("ARC_FilterWheel.dll Ver : %d.%d.%d \n", MajorVal, MinorVal, BuildVal);
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
     UnLoad_ARC_FilterWheel_dll();
     }
  else
  {
     printf("Error : DLL not Loaded!\n");
  }
  return 0;
}


