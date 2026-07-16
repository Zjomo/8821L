//34567891123456789212345678931234567894123456789512345678961234567897123456789
// ARC_FilterWheel_dll.h : Includes the necessary hooks for using the 
//                        Acton Research Corp. ARC_FilterWheel.dll
//
//#include <iostream>
//#include <fstream>
//#include <string>
//#include <ctype.h>
//#include <assert.h>
//#include <stdlib.h>
//#include <stdio.h>
//#include <conio.h>
#include <windows.h>
// function definitions
typedef unsigned _int16 (CALLBACK* LPFNDLL_Ver) (long &, long &, long &);
//Communications
typedef unsigned _int16 (CALLBACK* LPFNDLL_Search_For_Filter) (long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_Close_Filter) (long);
// FilterWheel
//Communications
typedef unsigned _int16 (CALLBACK* LPFNDLL_Open_Filter)       (long, long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_Open_Filter_Port)  (long, long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_Valid_Filter_Enum) (long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Filter_preOpen_Model_CString) (long, char &);

//Information
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Filter_Model_CString)  (long, char &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Filter_Serial_CString) (long, char &);

//Filter
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Filter_Present)  (long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Filter_Position) (long, long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_set_Filter_Position) (long, long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Filter_Min_Pos)  (long, long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Filter_Max_Pos)  (long, long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_Filter_Home) (long);

// function pointers
HINSTANCE hARC_FilterWheel; // Handle to ARC_FilterWheel.dll
LPFNDLL_Ver ARC_Ver;       // function pointer to ARC_Ver
//Communications
LPFNDLL_Search_For_Filter              ARC_Search_For_Filter;
LPFNDLL_Close_Filter                   ARC_Close_Filter;
// FilterWheel
//Communications
LPFNDLL_Open_Filter                    ARC_Open_Filter;
LPFNDLL_Open_Filter_Port               ARC_Open_Filter_Port;
LPFNDLL_Valid_Filter_Enum              ARC_Valid_Filter_Enum;
LPFNDLL_get_Filter_preOpen_Model_CString 
                                       ARC_get_Filter_preOpen_Model_CString;

//Information
LPFNDLL_get_Filter_Model_CString       ARC_get_Filter_Model_CString;
LPFNDLL_get_Filter_Serial_CString      ARC_get_Filter_Serial_CString;
//Filter
LPFNDLL_get_Filter_Present             ARC_get_Filter_Present;
LPFNDLL_get_Filter_Position            ARC_get_Filter_Position;
LPFNDLL_set_Filter_Position            ARC_set_Filter_Position;
LPFNDLL_get_Filter_Min_Pos             ARC_get_Filter_Min_Pos; 
LPFNDLL_get_Filter_Max_Pos             ARC_get_Filter_Max_Pos;
LPFNDLL_Filter_Home                    ARC_Filter_Home;

int Setup_ARC_FilterWheel_dll()
{
   hARC_FilterWheel = LoadLibrary("ARC_FilterWheel.dll");
   if (hARC_FilterWheel != NULL)
   {
    // 
    ARC_Ver                             = (LPFNDLL_Ver) GetProcAddress(hARC_FilterWheel,"ARC_Ver");
    //Communications
    ARC_Search_For_Filter               = (LPFNDLL_Search_For_Filter)  GetProcAddress(hARC_FilterWheel,"ARC_Search_For_Filter");
    ARC_Close_Filter                    = (LPFNDLL_Close_Filter)  GetProcAddress(hARC_FilterWheel,"ARC_Close_Filter");
    // FilterWheel
    //Communications
    ARC_Open_Filter                     = (LPFNDLL_Open_Filter)  GetProcAddress(hARC_FilterWheel,"ARC_Open_Filter");
    ARC_Open_Filter_Port                = (LPFNDLL_Open_Filter_Port)  GetProcAddress(hARC_FilterWheel,"ARC_Open_Filter_Port");
    ARC_Valid_Filter_Enum               = (LPFNDLL_Valid_Filter_Enum)  GetProcAddress(hARC_FilterWheel,"ARC_Valid_Filter_Enum");
    ARC_get_Filter_preOpen_Model_CString 
                                        = (LPFNDLL_get_Filter_preOpen_Model_CString)  GetProcAddress(hARC_FilterWheel,"ARC_get_Filter_preOpen_Model_CString");

    //Information
    ARC_get_Filter_Model_CString        = (LPFNDLL_get_Filter_Model_CString)  GetProcAddress(hARC_FilterWheel,"ARC_get_Filter_Model_CString");
    ARC_get_Filter_Serial_CString       = (LPFNDLL_get_Filter_Serial_CString)  GetProcAddress(hARC_FilterWheel,"ARC_get_Filter_Serial_CString");
    //Filter
    ARC_get_Filter_Present              = (LPFNDLL_get_Filter_Present)  GetProcAddress(hARC_FilterWheel,"ARC_get_Filter_Present");
    ARC_get_Filter_Position             = (LPFNDLL_get_Filter_Position)  GetProcAddress(hARC_FilterWheel,"ARC_get_Filter_Position");
    ARC_set_Filter_Position             = (LPFNDLL_set_Filter_Position)  GetProcAddress(hARC_FilterWheel,"ARC_set_Filter_Position");
    ARC_get_Filter_Min_Pos              = (LPFNDLL_get_Filter_Min_Pos)  GetProcAddress(hARC_FilterWheel,"ARC_get_Filter_Min_Pos"); 
    ARC_get_Filter_Max_Pos              = (LPFNDLL_get_Filter_Max_Pos)  GetProcAddress(hARC_FilterWheel,"ARC_get_Filter_Max_Pos");
    ARC_Filter_Home                     = (LPFNDLL_Filter_Home)  GetProcAddress(hARC_FilterWheel,"ARC_Filter_Home");

	if (ARC_Ver != NULL &&
       //Communications
	   ARC_Search_For_Filter              != NULL && 
       ARC_Close_Filter                   != NULL &&
       // FilterWheel
       //Communications
       ARC_Open_Filter                    != NULL && 
       ARC_Open_Filter_Port               != NULL && 
       ARC_Valid_Filter_Enum              != NULL && 
       ARC_get_Filter_preOpen_Model_CString 
                                       != NULL &&
       //Information
       ARC_get_Filter_Model_CString       != NULL && 
       ARC_get_Filter_Serial_CString      != NULL && 
       //Filter
       ARC_get_Filter_Present             != NULL && 
       ARC_get_Filter_Position            != NULL && 
       ARC_set_Filter_Position            != NULL && 
       ARC_get_Filter_Min_Pos             != NULL && 
       ARC_get_Filter_Max_Pos             != NULL && 
       ARC_Filter_Home                    != NULL ) 
    {
	return TRUE;
	}
	else
    {
	return FALSE;
	}
   }
   else
   {
    // Failed to load the DLL 
    return FALSE;	   
   }
}

void UnLoad_ARC_FilterWheel_dll()
{
// code here to unload any open ports
FreeLibrary(hARC_FilterWheel);
}