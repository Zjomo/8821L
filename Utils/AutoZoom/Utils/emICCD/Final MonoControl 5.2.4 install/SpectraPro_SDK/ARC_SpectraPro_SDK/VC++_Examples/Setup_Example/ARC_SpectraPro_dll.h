//34567891123456789212345678931234567894123456789512345678961234567897123456789
// ARC_SpectraPro_dll.h : Includes the necessary hooks for using the 
//                        Acton Research Corp. ARC_SpectraPro.dll
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
typedef unsigned _int16 (CALLBACK* LPFNDLL_Search_For_Mono) (long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_Open_Mono) (long,long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_Open_Mono_Port) (long, long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_Close_Mono) (long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_Valid_Mono_Enum) (long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_preOpen_Model_CString) (long, char &);

//Information
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Model_CString) (long, char &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Serial_CString) (long, char &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Focallength) (long, double &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_HalfAngle) (long, double &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_DetectorAngle) (long, double &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Double) (long, unsigned _int16 &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Precision) (long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Backlash_Steps) (long, long&);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Gear_Steps) (long, long &, long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_nmRev_Ratio) (long, double &);

//Wavelength
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Wavelength_nm) (long, double &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_set_Mono_Wavelength_nm) (long, double);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Wavelength_ang) (long, double &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_set_Mono_Wavelength_ang) (long, double);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Wavelength_eV) (long, double &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_set_Mono_Wavelength_eV) (long, double);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Wavelength_micron) (long, double &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_set_Mono_Wavelength_micron) (long, double);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Wavelength_absCM) (long, double &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_set_Mono_Wavelength_absCM) (long, double);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Wavelength_relCM) (long, double, double &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_set_Mono_Wavelength_relCM) (long, double, double);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Wavelength_Cutoff_nm) (long, double &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Wavelength_Min_nm) (long, double &);

//Grating
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Turret) (long, long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_set_Mono_Turret) (long, long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Turret_Gratings) (long, long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Grating) (long, long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_set_Mono_Grating) (long, long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Grating_Blaze_CString) (long, long, char &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Grating_Density) (long, long, long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Grating_Installed) (long, long);

//Mirrors
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Diverter_Valid) (long, long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Diverter_Pos) (long, long, long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_set_Mono_Diverter_Pos) (long, long, long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Diverter_Pos_CString) (long, long, char &);

//Slits
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Slit_Type) (long, long, long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Slit_Type_CString) (long, long, char &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Slit_Width) (long, long, long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_set_Mono_Slit_Width) (long, long, long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_Mono_Slit_Home) (long, long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_Mono_Slit_Name_CString) (long, char &);

//Filter
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Filter_Present) (long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Filter_Position) (long, long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_set_Mono_Filter_Position) (long, long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Filter_Min_Pos) (long, long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Filter_Max_Pos) (long, long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_Mono_Filter_Home) (long);

// advanced functions
// Gear
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Int_Led_On) (long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_set_Mono_Int_Led) (long, int);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Motor_Int) (long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Wheel_Int) (long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_Mono_Move_Steps) (long, long);

// Grating Values
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Init_Grating) (long, long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_set_Mono_Init_Grating) (long, long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Init_Wave_nm) (long, double &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_set_Mono_Init_Wave_nm) (long, double);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Init_ScanRate_nm) (long, double &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_set_Mono_Init_ScanRate_nm) (long, double);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Grating_Offset) (long, long, long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Grating_Gadjust) (long, long, long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_set_Mono_Grating_Offset) (long, long, long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_set_Mono_Grating_Gadjust) (long, long, long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_Mono_Grating_Calc_Offset) (long, long, double, double, long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_Mono_Grating_Calc_Gadjust) (long, long, double, double, long &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_Mono_Reset) (long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_Mono_Restore_Factory_Settings) (long);

// Scan Values
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Scan_Rate_nm_min)  (long, double &);
typedef unsigned _int16 (CALLBACK* LPFNDLL_set_Mono_Scan_Rate_nm_min)  (long, double);
typedef unsigned _int16 (CALLBACK* LPFNDLL_Mono_Start_Scan_To_nm)  (long, double);
typedef unsigned _int16 (CALLBACK* LPFNDLL_Mono_Scan_Done)  (long, long &, double);
typedef unsigned _int16 (CALLBACK* LPFNDLL_Mono_Start_Jog)  (long, long, long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_Mono_Stop_Jog)  (long);

// shutter control for v6 controllers
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Shutter_Valid)   (long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_get_Mono_Shutter_Open)    (long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_set_Mono_Shutter_Open)    (long);
typedef unsigned _int16 (CALLBACK* LPFNDLL_set_Mono_Shutter_Closed)  (long);

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
HINSTANCE hARC_SpectraPro; // Handle to ARC_SpectraPro.dll
LPFNDLL_Ver ARC_Ver;       // function pointer to ARC_Ver
//Communications
LPFNDLL_Search_For_Mono                ARC_Search_For_Mono;
LPFNDLL_Open_Mono                      ARC_Open_Mono;
LPFNDLL_Open_Mono_Port                 ARC_Open_Mono_Port;
LPFNDLL_Close_Mono                     ARC_Close_Mono;
LPFNDLL_Valid_Mono_Enum                ARC_Valid_Mono_Enum;
LPFNDLL_get_Mono_preOpen_Model_CString ARC_get_Mono_preOpen_Model_CString;

//Information
LPFNDLL_get_Mono_Model_CString         ARC_get_Mono_Model_CString;
LPFNDLL_get_Mono_Serial_CString        ARC_get_Mono_Serial_CString;
LPFNDLL_get_Mono_Focallength           ARC_get_Mono_Focallength;
LPFNDLL_get_Mono_HalfAngle             ARC_get_Mono_HalfAngle;
LPFNDLL_get_Mono_DetectorAngle         ARC_get_Mono_DetectorAngle;
LPFNDLL_get_Mono_Double                ARC_get_Mono_Double;
LPFNDLL_get_Mono_Precision             ARC_get_Mono_Precision;

//Wavelength
LPFNDLL_get_Mono_Wavelength_nm         ARC_get_Mono_Wavelength_nm;
LPFNDLL_set_Mono_Wavelength_nm         ARC_set_Mono_Wavelength_nm;
LPFNDLL_get_Mono_Wavelength_ang        ARC_get_Mono_Wavelength_ang;
LPFNDLL_set_Mono_Wavelength_ang        ARC_set_Mono_Wavelength_ang;
LPFNDLL_get_Mono_Wavelength_eV         ARC_get_Mono_Wavelength_eV;
LPFNDLL_set_Mono_Wavelength_eV         ARC_set_Mono_Wavelength_eV;
LPFNDLL_get_Mono_Wavelength_micron     ARC_get_Mono_Wavelength_micron;
LPFNDLL_set_Mono_Wavelength_micron     ARC_set_Mono_Wavelength_micron;
LPFNDLL_get_Mono_Wavelength_absCM      ARC_get_Mono_Wavelength_absCM;
LPFNDLL_set_Mono_Wavelength_absCM      ARC_set_Mono_Wavelength_absCM;
LPFNDLL_get_Mono_Wavelength_relCM      ARC_get_Mono_Wavelength_relCM;
LPFNDLL_set_Mono_Wavelength_relCM      ARC_set_Mono_Wavelength_relCM;
LPFNDLL_get_Mono_Wavelength_Cutoff_nm  ARC_get_Mono_Wavelength_Cutoff_nm;
LPFNDLL_get_Mono_Wavelength_Min_nm     ARC_get_Mono_Wavelength_Min_nm;

//Grating
LPFNDLL_get_Mono_Turret                ARC_get_Mono_Turret;
LPFNDLL_set_Mono_Turret                ARC_set_Mono_Turret;
LPFNDLL_get_Mono_Turret_Gratings       ARC_get_Mono_Turret_Gratings;
LPFNDLL_get_Mono_Grating               ARC_get_Mono_Grating;
LPFNDLL_set_Mono_Grating               ARC_set_Mono_Grating;
LPFNDLL_get_Mono_Grating_Blaze_CString ARC_get_Mono_Grating_Blaze_CString;
LPFNDLL_get_Mono_Grating_Density       ARC_get_Mono_Grating_Density;
LPFNDLL_get_Mono_Grating_Installed     ARC_get_Mono_Grating_Installed;

//Mirrors
LPFNDLL_get_Mono_Diverter_Valid        ARC_get_Mono_Diverter_Valid;
LPFNDLL_get_Mono_Diverter_Pos          ARC_get_Mono_Diverter_Pos;
LPFNDLL_set_Mono_Diverter_Pos          ARC_set_Mono_Diverter_Pos;
LPFNDLL_get_Mono_Diverter_Pos_CString  ARC_get_Mono_Diverter_Pos_CString;

//Slits
LPFNDLL_get_Mono_Slit_Type             ARC_get_Mono_Slit_Type;
LPFNDLL_get_Mono_Slit_Type_CString     ARC_get_Mono_Slit_Type_CString;
LPFNDLL_get_Mono_Slit_Width            ARC_get_Mono_Slit_Width;
LPFNDLL_set_Mono_Slit_Width            ARC_set_Mono_Slit_Width;
LPFNDLL_Mono_Slit_Home                 ARC_Mono_Slit_Home;
LPFNDLL_Mono_Slit_Name_CString         ARC_Mono_Slit_Name_CString;

//Filter
LPFNDLL_get_Mono_Filter_Present        ARC_get_Mono_Filter_Present;
LPFNDLL_get_Mono_Filter_Position       ARC_get_Mono_Filter_Position;
LPFNDLL_set_Mono_Filter_Position       ARC_set_Mono_Filter_Position;
LPFNDLL_get_Mono_Filter_Min_Pos        ARC_get_Mono_Filter_Min_Pos;
LPFNDLL_get_Mono_Filter_Max_Pos        ARC_get_Mono_Filter_Max_Pos;
LPFNDLL_Mono_Filter_Home               ARC_Mono_Filter_Home;

// advanced functions
// Gear
LPFNDLL_get_Mono_Int_Led_On            ARC_get_Mono_Int_Led_On;
LPFNDLL_set_Mono_Int_Led               ARC_set_Mono_Int_Led;
LPFNDLL_get_Mono_Motor_Int             ARC_get_Mono_Motor_Int;
LPFNDLL_get_Mono_Wheel_Int             ARC_get_Mono_Wheel_Int;
LPFNDLL_Mono_Move_Steps                ARC_Mono_Move_Steps;

// Grating Values
LPFNDLL_get_Mono_Init_Grating          ARC_get_Mono_Init_Grating;
LPFNDLL_set_Mono_Init_Grating          ARC_set_Mono_Init_Grating;
LPFNDLL_get_Mono_Init_Wave_nm          ARC_get_Mono_Init_Wave_nm;
LPFNDLL_set_Mono_Init_Wave_nm          ARC_set_Mono_Init_Wave_nm;
LPFNDLL_get_Mono_Init_ScanRate_nm      ARC_get_Mono_Init_ScanRate_nm;
LPFNDLL_set_Mono_Init_ScanRate_nm      ARC_set_Mono_Init_ScanRate_nm;
LPFNDLL_get_Mono_Grating_Offset        ARC_get_Mono_Grating_Offset;
LPFNDLL_get_Mono_Grating_Gadjust       ARC_get_Mono_Grating_Gadjust;
LPFNDLL_set_Mono_Grating_Offset        ARC_set_Mono_Grating_Offset;
LPFNDLL_set_Mono_Grating_Gadjust       ARC_set_Mono_Grating_Gadjust;
LPFNDLL_Mono_Grating_Calc_Offset       ARC_Mono_Grating_Calc_Offset;
LPFNDLL_Mono_Grating_Calc_Gadjust      ARC_Mono_Grating_Calc_Gadjust;
LPFNDLL_Mono_Reset                     ARC_Mono_Reset;
LPFNDLL_Mono_Restore_Factory_Settings  ARC_Mono_Restore_Factory_Settings;

// Scan Values
LPFNDLL_get_Mono_Scan_Rate_nm_min      ARC_get_Mono_Scan_Rate_nm_min; 
LPFNDLL_set_Mono_Scan_Rate_nm_min      ARC_set_Mono_Scan_Rate_nm_min; 
LPFNDLL_Mono_Start_Scan_To_nm          ARC_Mono_Start_Scan_To_nm;
LPFNDLL_Mono_Scan_Done                 ARC_Mono_Scan_Done; 
LPFNDLL_Mono_Start_Jog                 ARC_Mono_Start_Jog; 
LPFNDLL_Mono_Stop_Jog                  ARC_Mono_Stop_Jog; 

// shutter control for v6 controllers
LPFNDLL_get_Mono_Shutter_Valid         ARC_get_Mono_Shutter_Valid; 
LPFNDLL_get_Mono_Shutter_Open          ARC_get_Mono_Shutter_Open; 
LPFNDLL_set_Mono_Shutter_Open          ARC_set_Mono_Shutter_Open; 
LPFNDLL_set_Mono_Shutter_Closed        ARC_set_Mono_Shutter_Closed;

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

int Setup_ARC_SpectraPro_dll()
{
   hARC_SpectraPro = LoadLibrary("ARC_SpectraPro.dll");
   if (hARC_SpectraPro != NULL)
   {
    // 
    ARC_Ver                             = (LPFNDLL_Ver) GetProcAddress(hARC_SpectraPro,"ARC_Ver");
    //Communications
    ARC_Search_For_Mono                 = (LPFNDLL_Search_For_Mono)  GetProcAddress(hARC_SpectraPro,"ARC_Search_For_Mono");
    ARC_Open_Mono                       = (LPFNDLL_Open_Mono)  GetProcAddress(hARC_SpectraPro,"ARC_Open_Mono");
    ARC_Open_Mono_Port                  = (LPFNDLL_Open_Mono_Port)  GetProcAddress(hARC_SpectraPro,"ARC_Open_Mono_Port");
    ARC_Close_Mono                      = (LPFNDLL_Close_Mono)  GetProcAddress(hARC_SpectraPro,"ARC_Close_Mono");
    ARC_Valid_Mono_Enum                 = (LPFNDLL_Valid_Mono_Enum)  GetProcAddress(hARC_SpectraPro,"ARC_Valid_Mono_Enum");
    ARC_get_Mono_preOpen_Model_CString  = (LPFNDLL_get_Mono_preOpen_Model_CString)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_preOpen_Model_CString");
    //Information
    ARC_get_Mono_Model_CString          = (LPFNDLL_get_Mono_Model_CString)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Model_CString");
    ARC_get_Mono_Serial_CString         = (LPFNDLL_get_Mono_Serial_CString)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Serial_CString");
    ARC_get_Mono_Focallength            = (LPFNDLL_get_Mono_Focallength)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Focallength");
    ARC_get_Mono_HalfAngle              = (LPFNDLL_get_Mono_HalfAngle)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_HalfAngle");
    ARC_get_Mono_DetectorAngle          = (LPFNDLL_get_Mono_DetectorAngle)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_DetectorAngle");
    ARC_get_Mono_Double                 = (LPFNDLL_get_Mono_Double)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Double");
    ARC_get_Mono_Precision              = (LPFNDLL_get_Mono_Precision)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Precision");

    //Wavelength
    ARC_get_Mono_Wavelength_nm          = (LPFNDLL_get_Mono_Wavelength_nm)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Wavelength_nm");
    ARC_set_Mono_Wavelength_nm          = (LPFNDLL_set_Mono_Wavelength_nm)  GetProcAddress(hARC_SpectraPro,"ARC_set_Mono_Wavelength_nm");
    ARC_get_Mono_Wavelength_ang         = (LPFNDLL_get_Mono_Wavelength_ang)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Wavelength_ang");
    ARC_set_Mono_Wavelength_ang         = (LPFNDLL_set_Mono_Wavelength_ang)  GetProcAddress(hARC_SpectraPro,"ARC_set_Mono_Wavelength_ang");
    ARC_get_Mono_Wavelength_eV          = (LPFNDLL_get_Mono_Wavelength_eV)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Wavelength_eV");
    ARC_set_Mono_Wavelength_eV          = (LPFNDLL_set_Mono_Wavelength_eV)  GetProcAddress(hARC_SpectraPro,"ARC_set_Mono_Wavelength_eV");
    ARC_get_Mono_Wavelength_micron      = (LPFNDLL_get_Mono_Wavelength_micron)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Wavelength_micron");
    ARC_set_Mono_Wavelength_micron      = (LPFNDLL_set_Mono_Wavelength_micron)  GetProcAddress(hARC_SpectraPro,"ARC_set_Mono_Wavelength_micron");
    ARC_get_Mono_Wavelength_absCM       = (LPFNDLL_get_Mono_Wavelength_absCM)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Wavelength_absCM");
    ARC_set_Mono_Wavelength_absCM       = (LPFNDLL_set_Mono_Wavelength_absCM)  GetProcAddress(hARC_SpectraPro,"ARC_set_Mono_Wavelength_absCM");
    ARC_get_Mono_Wavelength_relCM       = (LPFNDLL_get_Mono_Wavelength_relCM)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Wavelength_relCM");
    ARC_set_Mono_Wavelength_relCM       = (LPFNDLL_set_Mono_Wavelength_relCM)  GetProcAddress(hARC_SpectraPro,"ARC_set_Mono_Wavelength_relCM");
    ARC_get_Mono_Wavelength_Cutoff_nm   = (LPFNDLL_get_Mono_Wavelength_Cutoff_nm)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Wavelength_Cutoff_nm");
    ARC_get_Mono_Wavelength_Min_nm      = (LPFNDLL_get_Mono_Wavelength_Min_nm)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Wavelength_Min_nm");

    //Grating
    ARC_get_Mono_Turret                 = (LPFNDLL_get_Mono_Turret)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Turret");
    ARC_set_Mono_Turret                 = (LPFNDLL_set_Mono_Turret)  GetProcAddress(hARC_SpectraPro,"ARC_set_Mono_Turret");
    ARC_get_Mono_Turret_Gratings        = (LPFNDLL_get_Mono_Turret_Gratings)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Turret_Gratings");
    ARC_get_Mono_Grating                = (LPFNDLL_get_Mono_Grating)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Grating");
    ARC_set_Mono_Grating                = (LPFNDLL_set_Mono_Grating)  GetProcAddress(hARC_SpectraPro,"ARC_set_Mono_Grating");
    ARC_get_Mono_Grating_Blaze_CString  = (LPFNDLL_get_Mono_Grating_Blaze_CString)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Grating_Blaze_CString");
    ARC_get_Mono_Grating_Density        = (LPFNDLL_get_Mono_Grating_Density)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Grating_Density");
    ARC_get_Mono_Grating_Installed      = (LPFNDLL_get_Mono_Grating_Installed)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Grating_Installed");

    //Mirrors
    ARC_get_Mono_Diverter_Valid         = (LPFNDLL_get_Mono_Diverter_Valid)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Diverter_Valid");
    ARC_get_Mono_Diverter_Pos           = (LPFNDLL_get_Mono_Diverter_Pos)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Diverter_Pos");
    ARC_set_Mono_Diverter_Pos           = (LPFNDLL_set_Mono_Diverter_Pos)  GetProcAddress(hARC_SpectraPro,"ARC_set_Mono_Diverter_Pos");
    ARC_get_Mono_Diverter_Pos_CString   = (LPFNDLL_get_Mono_Diverter_Pos_CString)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Diverter_Pos_CString");

    //Slits
    ARC_get_Mono_Slit_Type              = (LPFNDLL_get_Mono_Slit_Type)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Slit_Type");
    ARC_get_Mono_Slit_Type_CString      = (LPFNDLL_get_Mono_Slit_Type_CString)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Slit_Type_CString");
    ARC_get_Mono_Slit_Width             = (LPFNDLL_get_Mono_Slit_Width)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Slit_Width");
    ARC_set_Mono_Slit_Width             = (LPFNDLL_set_Mono_Slit_Width)  GetProcAddress(hARC_SpectraPro,"ARC_set_Mono_Slit_Width");
    ARC_Mono_Slit_Home                  = (LPFNDLL_Mono_Slit_Home)  GetProcAddress(hARC_SpectraPro,"ARC_Mono_Slit_Home");
    ARC_Mono_Slit_Name_CString          = (LPFNDLL_Mono_Slit_Name_CString)  GetProcAddress(hARC_SpectraPro,"ARC_Mono_Slit_Name_CString");

    //Filter
    ARC_get_Mono_Filter_Present         = (LPFNDLL_get_Mono_Filter_Present)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Filter_Present");
    ARC_get_Mono_Filter_Position        = (LPFNDLL_get_Mono_Filter_Position)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Filter_Position");
    ARC_set_Mono_Filter_Position        = (LPFNDLL_set_Mono_Filter_Position)  GetProcAddress(hARC_SpectraPro,"ARC_set_Mono_Filter_Position");
    ARC_get_Mono_Filter_Min_Pos         = (LPFNDLL_get_Mono_Filter_Min_Pos)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Filter_Min_Pos");
    ARC_get_Mono_Filter_Max_Pos         = (LPFNDLL_get_Mono_Filter_Max_Pos)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Filter_Max_Pos");
    ARC_Mono_Filter_Home                = (LPFNDLL_Mono_Filter_Home)  GetProcAddress(hARC_SpectraPro,"ARC_Mono_Filter_Home");

    // advanced functions
    // Gear
    ARC_get_Mono_Int_Led_On             = (LPFNDLL_get_Mono_Int_Led_On)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Int_Led_On");
    ARC_set_Mono_Int_Led                = (LPFNDLL_set_Mono_Int_Led)  GetProcAddress(hARC_SpectraPro,"ARC_set_Mono_Int_Led");
    ARC_get_Mono_Motor_Int              = (LPFNDLL_get_Mono_Motor_Int)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Motor_Int");
    ARC_get_Mono_Wheel_Int              = (LPFNDLL_get_Mono_Wheel_Int)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Wheel_Int");
    ARC_Mono_Move_Steps                 = (LPFNDLL_Mono_Move_Steps)  GetProcAddress(hARC_SpectraPro,"ARC_Mono_Move_Steps");

    // Grating Values
    ARC_get_Mono_Init_Grating           = (LPFNDLL_get_Mono_Init_Grating)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Init_Grating");
    ARC_set_Mono_Init_Grating           = (LPFNDLL_set_Mono_Init_Grating)  GetProcAddress(hARC_SpectraPro,"ARC_set_Mono_Init_Grating");
    ARC_get_Mono_Init_Wave_nm           = (LPFNDLL_get_Mono_Init_Wave_nm)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Init_Wave_nm");
    ARC_set_Mono_Init_Wave_nm           = (LPFNDLL_set_Mono_Init_Wave_nm)  GetProcAddress(hARC_SpectraPro,"ARC_set_Mono_Init_Wave_nm");
    ARC_get_Mono_Init_ScanRate_nm       = (LPFNDLL_get_Mono_Init_ScanRate_nm)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Init_ScanRate_nm");
    ARC_set_Mono_Init_ScanRate_nm       = (LPFNDLL_set_Mono_Init_ScanRate_nm)  GetProcAddress(hARC_SpectraPro,"ARC_set_Mono_Init_ScanRate_nm");
    ARC_get_Mono_Grating_Offset         = (LPFNDLL_get_Mono_Grating_Offset)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Grating_Offset");
    ARC_get_Mono_Grating_Gadjust        = (LPFNDLL_get_Mono_Grating_Gadjust)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Grating_Gadjust");
    ARC_set_Mono_Grating_Offset         = (LPFNDLL_set_Mono_Grating_Offset)  GetProcAddress(hARC_SpectraPro,"ARC_set_Mono_Grating_Offset");
    ARC_set_Mono_Grating_Gadjust        = (LPFNDLL_set_Mono_Grating_Gadjust)  GetProcAddress(hARC_SpectraPro,"ARC_set_Mono_Grating_Gadjust");
    ARC_Mono_Grating_Calc_Offset        = (LPFNDLL_Mono_Grating_Calc_Offset)  GetProcAddress(hARC_SpectraPro,"ARC_Mono_Grating_Calc_Offset");
    ARC_Mono_Grating_Calc_Gadjust       = (LPFNDLL_Mono_Grating_Calc_Gadjust)  GetProcAddress(hARC_SpectraPro,"ARC_Mono_Grating_Calc_Gadjust");
    ARC_Mono_Reset                      = (LPFNDLL_Mono_Reset)  GetProcAddress(hARC_SpectraPro,"ARC_Mono_Reset");
    ARC_Mono_Restore_Factory_Settings   = (LPFNDLL_Mono_Restore_Factory_Settings)  GetProcAddress(hARC_SpectraPro,"ARC_Mono_Restore_Factory_Settings");

    // Scan Values
    ARC_get_Mono_Scan_Rate_nm_min       = (LPFNDLL_get_Mono_Scan_Rate_nm_min)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Scan_Rate_nm_min"); 
    ARC_set_Mono_Scan_Rate_nm_min       = (LPFNDLL_set_Mono_Scan_Rate_nm_min)  GetProcAddress(hARC_SpectraPro,"ARC_set_Mono_Scan_Rate_nm_min"); 
    ARC_Mono_Start_Scan_To_nm           = (LPFNDLL_Mono_Start_Scan_To_nm)  GetProcAddress(hARC_SpectraPro,"ARC_Mono_Start_Scan_To_nm");
    ARC_Mono_Scan_Done                  = (LPFNDLL_Mono_Scan_Done)  GetProcAddress(hARC_SpectraPro,"ARC_Mono_Scan_Done"); 
    ARC_Mono_Start_Jog                  = (LPFNDLL_Mono_Start_Jog)  GetProcAddress(hARC_SpectraPro,"ARC_Mono_Start_Jog"); 
    ARC_Mono_Stop_Jog                   = (LPFNDLL_Mono_Stop_Jog)  GetProcAddress(hARC_SpectraPro,"ARC_Mono_Stop_Jog"); 

    // shutter control for v6 controllers
    ARC_get_Mono_Shutter_Valid          = (LPFNDLL_get_Mono_Shutter_Valid)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Shutter_Valid"); 
    ARC_get_Mono_Shutter_Open           = (LPFNDLL_get_Mono_Shutter_Open)  GetProcAddress(hARC_SpectraPro,"ARC_get_Mono_Shutter_Open"); 
    ARC_set_Mono_Shutter_Open           = (LPFNDLL_set_Mono_Shutter_Open)  GetProcAddress(hARC_SpectraPro,"ARC_set_Mono_Shutter_Open"); 
    ARC_set_Mono_Shutter_Closed         = (LPFNDLL_set_Mono_Shutter_Closed)  GetProcAddress(hARC_SpectraPro,"ARC_set_Mono_Shutter_Closed");

    // FilterWheel
    //Communications
    ARC_Open_Filter                     = (LPFNDLL_Open_Filter)  GetProcAddress(hARC_SpectraPro,"ARC_Open_Filter");
    ARC_Open_Filter_Port                = (LPFNDLL_Open_Filter_Port)  GetProcAddress(hARC_SpectraPro,"ARC_Open_Filter_Port");
    ARC_Valid_Filter_Enum               = (LPFNDLL_Valid_Filter_Enum)  GetProcAddress(hARC_SpectraPro,"ARC_Valid_Filter_Enum");
    ARC_get_Filter_preOpen_Model_CString 
                                        = (LPFNDLL_get_Filter_preOpen_Model_CString)  GetProcAddress(hARC_SpectraPro,"ARC_get_Filter_preOpen_Model_CString");

    //Information
    ARC_get_Filter_Model_CString        = (LPFNDLL_get_Filter_Model_CString)  GetProcAddress(hARC_SpectraPro,"ARC_get_Filter_Model_CString");
    ARC_get_Filter_Serial_CString       = (LPFNDLL_get_Filter_Serial_CString)  GetProcAddress(hARC_SpectraPro,"ARC_get_Filter_Serial_CString");
    //Filter
    ARC_get_Filter_Present              = (LPFNDLL_get_Filter_Present)  GetProcAddress(hARC_SpectraPro,"ARC_get_Filter_Present");
    ARC_get_Filter_Position             = (LPFNDLL_get_Filter_Position)  GetProcAddress(hARC_SpectraPro,"ARC_get_Filter_Position");
    ARC_set_Filter_Position             = (LPFNDLL_set_Filter_Position)  GetProcAddress(hARC_SpectraPro,"ARC_set_Filter_Position");
    ARC_get_Filter_Min_Pos              = (LPFNDLL_get_Filter_Min_Pos)  GetProcAddress(hARC_SpectraPro,"ARC_get_Filter_Min_Pos"); 
    ARC_get_Filter_Max_Pos              = (LPFNDLL_get_Filter_Max_Pos)  GetProcAddress(hARC_SpectraPro,"ARC_get_Filter_Max_Pos");
    ARC_Filter_Home                     = (LPFNDLL_Filter_Home)  GetProcAddress(hARC_SpectraPro,"ARC_Filter_Home");

	if (ARC_Ver != NULL &&
       //Communications
	   ARC_Search_For_Mono                != NULL && 
       ARC_Open_Mono                      != NULL && 
       ARC_Open_Mono_Port                 != NULL && 
       ARC_Close_Mono                     != NULL && 
       ARC_Valid_Mono_Enum                != NULL && 
       ARC_get_Mono_preOpen_Model_CString != NULL && 
       //Information
       ARC_get_Mono_Model_CString         != NULL && 
       ARC_get_Mono_Serial_CString        != NULL && 
       ARC_get_Mono_Focallength           != NULL && 
       ARC_get_Mono_HalfAngle             != NULL && 
       ARC_get_Mono_DetectorAngle         != NULL && 
       ARC_get_Mono_Double                != NULL && 
       ARC_get_Mono_Precision             != NULL && 

       //Wavelength
       ARC_get_Mono_Wavelength_nm         != NULL && 
       ARC_set_Mono_Wavelength_nm         != NULL && 
       ARC_get_Mono_Wavelength_ang        != NULL && 
       ARC_set_Mono_Wavelength_ang        != NULL && 
       ARC_get_Mono_Wavelength_eV         != NULL && 
       ARC_set_Mono_Wavelength_eV         != NULL && 
       ARC_get_Mono_Wavelength_micron     != NULL && 
       ARC_set_Mono_Wavelength_micron     != NULL && 
       ARC_get_Mono_Wavelength_absCM      != NULL && 
       ARC_set_Mono_Wavelength_absCM      != NULL && 
       ARC_get_Mono_Wavelength_relCM      != NULL && 
       ARC_set_Mono_Wavelength_relCM      != NULL && 
       ARC_get_Mono_Wavelength_Cutoff_nm  != NULL && 
       ARC_get_Mono_Wavelength_Min_nm     != NULL && 

       //Grating
       ARC_get_Mono_Turret                != NULL && 
       ARC_set_Mono_Turret                != NULL && 
       ARC_get_Mono_Turret_Gratings       != NULL && 
       ARC_get_Mono_Grating               != NULL && 
       ARC_set_Mono_Grating               != NULL && 
       ARC_get_Mono_Grating_Blaze_CString != NULL && 
       ARC_get_Mono_Grating_Density       != NULL && 
       ARC_get_Mono_Grating_Installed     != NULL && 

       //Mirrors
       ARC_get_Mono_Diverter_Valid        != NULL && 
       ARC_get_Mono_Diverter_Pos          != NULL && 
       ARC_set_Mono_Diverter_Pos          != NULL && 
       ARC_get_Mono_Diverter_Pos_CString  != NULL && 

       //Slits
       ARC_get_Mono_Slit_Type             != NULL && 
       ARC_get_Mono_Slit_Type_CString     != NULL && 
       ARC_get_Mono_Slit_Width            != NULL && 
       ARC_set_Mono_Slit_Width            != NULL && 
       ARC_Mono_Slit_Home                 != NULL && 
       ARC_Mono_Slit_Name_CString         != NULL &&  

       //Filter
       ARC_get_Mono_Filter_Present        != NULL && 
       ARC_get_Mono_Filter_Position       != NULL && 
       ARC_set_Mono_Filter_Position       != NULL && 
       ARC_get_Mono_Filter_Min_Pos        != NULL && 
       ARC_get_Mono_Filter_Max_Pos        != NULL && 
       ARC_Mono_Filter_Home               != NULL && 

       // advanced functions
       // Gear
       ARC_get_Mono_Int_Led_On            != NULL &&
       ARC_set_Mono_Int_Led               != NULL &&
       ARC_get_Mono_Motor_Int             != NULL &&
       ARC_get_Mono_Wheel_Int             != NULL &&
       ARC_Mono_Move_Steps                != NULL &&

       // Grating Values
       ARC_get_Mono_Init_Grating          != NULL && 
       ARC_set_Mono_Init_Grating          != NULL &&
       ARC_get_Mono_Init_Wave_nm          != NULL &&
       ARC_set_Mono_Init_Wave_nm          != NULL &&
       ARC_get_Mono_Init_ScanRate_nm      != NULL &&
       ARC_set_Mono_Init_ScanRate_nm      != NULL &&
       ARC_get_Mono_Grating_Offset        != NULL &&
       ARC_get_Mono_Grating_Gadjust       != NULL &&
       ARC_set_Mono_Grating_Offset        != NULL &&
       ARC_set_Mono_Grating_Gadjust       != NULL &&
       ARC_Mono_Grating_Calc_Offset       != NULL &&
       ARC_Mono_Grating_Calc_Gadjust      != NULL &&
       ARC_Mono_Reset                     != NULL &&
       ARC_Mono_Restore_Factory_Settings  != NULL &&

       // Scan Values
       ARC_get_Mono_Scan_Rate_nm_min      != NULL &&
       ARC_set_Mono_Scan_Rate_nm_min      != NULL && 
       ARC_Mono_Start_Scan_To_nm          != NULL && 
       ARC_Mono_Scan_Done                 != NULL && 
       ARC_Mono_Start_Jog                 != NULL && 
       ARC_Mono_Stop_Jog                  != NULL && 

       // shutter control for v6 controllers
       ARC_get_Mono_Shutter_Valid         != NULL && 
       ARC_get_Mono_Shutter_Open          != NULL && 
       ARC_set_Mono_Shutter_Open          != NULL && 
       ARC_set_Mono_Shutter_Closed        != NULL && 

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

void UnLoad_ARC_SpectraPro_dll()
{
// code here to unload any open ports
FreeLibrary(hARC_SpectraPro);
}