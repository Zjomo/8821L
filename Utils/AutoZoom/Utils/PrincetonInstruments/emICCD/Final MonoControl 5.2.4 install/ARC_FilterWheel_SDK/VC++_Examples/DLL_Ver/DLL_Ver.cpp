// DLL_Ver.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ARC_FilterWheel_dll.h"

int main(int argc, char* argv[])
{
long MajorVal,MinorVal,BuildVal; ///return values from ARC_Ver

  if (Setup_ARC_FilterWheel_dll() == TRUE)
  {
  printf("DLL Loaded!\n");
  if (ARC_Ver(MajorVal,MinorVal,BuildVal) != 0)
  {
  printf("ARC_FilterWheel.dll Ver : %d.%d.%d \n", MajorVal, MinorVal, BuildVal);
  }
  else
  {
  printf("ARC_Ver Failed!\n");
  }
  UnLoad_ARC_FilterWheel_dll();
  }
  else
  {
  printf("DLL not Loaded!\n");
  }
  return 0;
}

