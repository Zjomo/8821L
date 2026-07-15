unit ARC_SpectraProU;

interface

uses
   windows,variants;

// dynamic load functions
function Dynamic_Load_ARC_SpectraPro_DLL : boolean;
procedure unload_ARC_SpectraPro_DLL;

//Communications
function ARC_Search_For_Mono(out Num_Found : integer): wordbool;
function ARC_Open_Mono(Enum_Num : integer; out Mono_Enum : integer): wordbool;
function ARC_Open_Mono_Port(Com_Num : integer; out Mono_Enum : integer) : wordbool;
function ARC_Close_Mono(Mono_Enum : integer): wordbool;  
function ARC_Valid_Mono_Enum(Mono_Enum : integer): wordbool;  
function ARC_get_Mono_preOpen_Model(Enum_Num : integer; out Model_str : string) : wordbool; 
function ARC_get_Mono_preOpen_Model_int32(Enum_Num : integer; out Model_int32 : integer) : wordbool;
function ARC_Ver(out Major : integer; out Minor : integer; out Build : integer) : wordbool;
// direct communications function
function ARC_Send_CMD_To_Mono(Mono_Enum : integer; SendCMD : string; out RCV_str : string; ms_Timeout : integer) : wordbool;  

//Information
function ARC_get_Mono_Model(Mono_Enum : integer;out Model_str : string): wordbool;  
function ARC_get_Mono_Serial(Mono_Enum : integer;out Serial_str : string): wordbool;  
function ARC_get_Mono_Focallength(Mono_Enum : integer; out Focallength : double): wordbool;  
function ARC_get_Mono_HalfAngle(Mono_Enum : integer; out HalfAngle : double): wordbool;  
function ARC_get_Mono_DetectorAngle(Mono_Enum : integer; out DetectorAngle : double): wordbool;  
function ARC_get_Mono_Double(Mono_Enum : integer;out Double_Present : wordbool): wordbool;         
function ARC_get_Mono_Double_Subtractive(Mono_Enum : integer;out Double_Subtractive : wordbool): wordbool;
function ARC_get_Mono_Precision(Mono_Enum : integer): integer;

// CCD Pixel Mapping
function ARC_get_Mono_Pixel_Map_nm    (Mono_Enum : integer; Pixel_Size_um : double; Number_X_Pixels : integer; Left_Adj_Percent : double; Center_Adj_Pixels : double; Right_Adj_Percent : double; Pixel_Map : pointer) : wordbool;
function ARC_get_Mono_Pixel_Map_ang   (Mono_Enum : integer; Pixel_Size_um : double; Number_X_Pixels : integer; Left_Adj_Percent : double; Center_Adj_Pixels : double; Right_Adj_Percent : double; Pixel_Map : pointer) : wordbool;
function ARC_get_Mono_Pixel_Map_eV    (Mono_Enum : integer; Pixel_Size_um : double; Number_X_Pixels : integer; Left_Adj_Percent : double; Center_Adj_Pixels : double; Right_Adj_Percent : double; Pixel_Map : pointer) : wordbool;
function ARC_get_Mono_Pixel_Map_micron(Mono_Enum : integer; Pixel_Size_um : double; Number_X_Pixels : integer; Left_Adj_Percent : double; Center_Adj_Pixels : double; Right_Adj_Percent : double; Pixel_Map : pointer) : wordbool;
function ARC_get_Mono_Pixel_Map_absCM (Mono_Enum : integer; Pixel_Size_um : double; Number_X_Pixels : integer; Left_Adj_Percent : double; Center_Adj_Pixels : double; Right_Adj_Percent : double; Pixel_Map : pointer) : wordbool;
function ARC_get_Mono_Pixel_Map_relCM (Mono_Enum : integer; Pixel_Size_um : double; Number_X_Pixels : integer; Laser_Line_nm : double; Left_Adj_Percent : double; Center_Adj_Pixels : double; Right_Adj_Percent : double; Pixel_Map : pointer) : wordbool; 

//Wavelength
function ARC_get_Mono_Wavelength_nm(Mono_Enum : integer; out Wavelength : double): wordbool;  
function ARC_set_Mono_Wavelength_nm(Mono_Enum : integer; Wavelength : double): wordbool;
function ARC_get_Mono_Wavelength_ang(Mono_Enum : integer; out Wavelength : double): wordbool;  
function ARC_set_Mono_Wavelength_ang(Mono_Enum : integer; Wavelength : double): wordbool;
function ARC_get_Mono_Wavelength_eV(Mono_Enum : integer; out Wavelength : double): wordbool;
function ARC_set_Mono_Wavelength_eV(Mono_Enum : integer; Wavelength : double): wordbool;  
function ARC_get_Mono_Wavelength_micron(Mono_Enum : integer; out Wavelength : double): wordbool;  
function ARC_set_Mono_Wavelength_micron(Mono_Enum : integer; Wavelength : double): wordbool;  
function ARC_get_Mono_Wavelength_absCM(Mono_Enum : integer; out Wavelength : double): wordbool;  
function ARC_set_Mono_Wavelength_absCM(Mono_Enum : integer; Wavelength : double): wordbool;  
function ARC_get_Mono_Wavelength_relCM(Mono_Enum : integer; Center_nm : double; out Wavelength : double): wordbool;  
function ARC_set_Mono_Wavelength_relCM(Mono_Enum : integer; Center_nm : double; Wavelength : double): wordbool;  
function ARC_get_Mono_Wavelength_Cutoff_nm(Mono_Enum : integer; out Wavelength : double): wordbool;  
function ARC_get_Mono_Wavelength_Min_nm(Mono_Enum : integer; out Wavelength : double): wordbool;  

//Grating
function ARC_get_Mono_Turret(Mono_Enum : integer; out Turret : integer): wordbool;  
function ARC_set_Mono_Turret(Mono_Enum : integer; Turret : integer): wordbool;  
function ARC_get_Mono_Turret_Gratings(Mono_Enum : integer; out Gratings_Per_Turret : integer): wordbool; 
function ARC_get_Mono_Turret_Max(Mono_Enum : integer; out Max_Turret_Number : integer): wordbool;
function ARC_get_Mono_Grating(Mono_Enum : integer; out Grating : integer): wordbool;  
function ARC_set_Mono_Grating(Mono_Enum : integer; Grating : integer): wordbool;  
function ARC_get_Mono_Grating_Blaze(Mono_Enum : integer; Grating : integer; out Blaze_str : string): wordbool;  
function ARC_get_Mono_Grating_Density(Mono_Enum : integer; Grating : integer; out Groove_MM : integer): wordbool;  
function ARC_get_Mono_Grating_Installed(Mono_Enum : integer; Grating : integer) : wordbool;  
function ARC_get_Mono_Grating_Max(Mono_Enum : integer; out Max_Grating_Number : integer): wordbool;

//Mirrors
function ARC_get_Mono_Diverter_Valid(Mono_Enum : integer; Diverter_Num : integer): wordbool;  
function ARC_get_Mono_Diverter_Pos(Mono_Enum : integer; Diverter_Num : integer; out Diverter_Pos : integer): wordbool;  
function ARC_set_Mono_Diverter_Pos(Mono_Enum : integer; Diverter_Num : integer; Diverter_Pos : integer): wordbool;
function ARC_get_Mono_Diverter_Pos_str(Mono_Enum : integer; Diverter_Num : integer; out Diverter_Pos : string): wordbool;  

//Slits
function ARC_get_Mono_Slit_Type(Mono_Enum : integer; Slit_Pos : integer; out Slit_Type : integer): wordbool;
function ARC_get_Mono_Slit_Type_str(Mono_Enum : integer; Slit_Pos : integer; out Slit_Type : string): wordbool;  
function ARC_get_Mono_Slit_Width(Mono_Enum : integer; Slit_Pos : integer; out Slit_Width : integer): wordbool;  
function ARC_set_Mono_Slit_Width(Mono_Enum : integer; Slit_Pos : integer; Slit_Width : integer): wordbool;
function ARC_Mono_Slit_Home(Mono_Enum : integer; Slit_Pos : integer): wordbool;
function ARC_Mono_Slit_Name_str(Slit_Num : integer; out Slit_Name_str : string) : wordbool;   
function ARC_Calc_Mono_Slit_BandPass(Mono_Enum : integer; Slit_Pos : integer; SlitWidth : double; out BandPass : double): wordbool; stdcall;
function ARC_get_Mono_Slit_BandPass(Mono_Enum : integer; Slit_Pos : integer; out BandPass : double): wordbool; stdcall;
function ARC_set_Mono_Slit_BandPass(Mono_ENum : integer; Slit_Pos : integer; BandPass : double): wordbool; stdcall;
// Special Double Function
function ARC_get_Mono_Double_Intermediate_Slit(Mono_Enum : integer; out Slit_Pos : integer): wordbool;
function ARC_get_Mono_Exit_Slit(Mono_Enum : integer; out Slit_Pos : integer): wordbool;

//Filter
function ARC_get_Mono_Filter_Present(Mono_Enum : integer): wordbool;
function ARC_get_Mono_Filter_Position(Mono_Enum : integer; out Position : integer): wordbool;
function ARC_set_Mono_Filter_Position(Mono_Enum : integer; Position : integer): wordbool;
function ARC_get_Mono_Filter_Min_Pos(Mono_Enum : integer; out Position : integer): wordbool;
function ARC_get_Mono_Filter_Max_Pos(Mono_Enum : integer; out Position : integer): wordbool;
function ARC_Mono_Filter_Home(Mono_Enum : integer): wordbool;

// advanced functions
// Gear
function ARC_get_Mono_Int_Led_On(Mono_Enum : integer): wordbool;
function ARC_set_Mono_Int_Led(Mono_Enum : integer; Led_State : wordbool): wordbool;
function ARC_get_Mono_Motor_Int(Mono_Enum : integer): wordbool;
function ARC_get_Mono_Wheel_Int(Mono_Enum : integer): wordbool;
function ARC_Mono_Move_Steps(Mono_Enum : integer; Num_Steps : integer): wordbool;

// Grating Values
function ARC_get_Mono_Init_Grating(Mono_Enum : integer; out Init_Grating : integer): wordbool;
function ARC_set_Mono_Init_Grating(Mono_Enum : integer; Init_Grating : integer): wordbool;
function ARC_get_Mono_Init_Wave_nm(Mono_Enum : integer; out Init_Wave : double): wordbool;
function ARC_set_Mono_Init_Wave_nm(Mono_Enum : integer; Init_Wave : double): wordbool;
function ARC_get_Mono_Init_ScanRate_nm(Mono_Enum : integer; out Init_ScanRate : double): wordbool;
function ARC_set_Mono_Init_ScanRate_nm(Mono_Enum : integer;  Init_ScanRate : double): wordbool;
function ARC_get_Mono_Grating_Offset(Mono_Enum : integer; Grating : integer; out Offset : integer): wordbool;
function ARC_get_Mono_Grating_Gadjust(Mono_Enum : integer; Grating : integer; out Gadjust : integer): wordbool;
function ARC_set_Mono_Grating_Offset(Mono_Enum : integer; Grating : integer; Offset : integer): wordbool;
function ARC_set_Mono_Grating_Gadjust(Mono_Enum : integer; Grating : integer; Gadjust : integer): wordbool;
function ARC_Mono_Grating_Calc_Offset(Mono_Enum : integer; Grating : integer; Wave : double; RefWave : double; out newOffset : integer): wordbool;
function ARC_Mono_Grating_Calc_Gadjust(Mono_Enum : integer; Grating : integer; Wave : double; RefWave : double; out newGadjust : integer): wordbool; 
function ARC_Mono_Grating_UnInstall(Mono_Enum : integer; Grating : integer): wordbool;
function ARC_Mono_Grating_Install(Mono_Enum : integer; Grating : integer; Density : Integer; Blaze : OleVariant; NMBlaze : boolean; HOLBlaze : boolean; MIRROR : boolean): wordbool;
function ARC_Mono_Reset(Mono_Enum : integer): wordbool;
function ARC_Mono_Restore_Factory_Settings(Mono_Enum : integer) : wordbool;
function ARC_Mono_Save_Factory_Settings(Mono_Enum : integer; password : double) : wordbool;

// Scan Values
function ARC_get_Mono_Scan_Rate_nm_min (Mono_Enum : integer; out Scan_Rate : double): wordbool;
function ARC_set_Mono_Scan_Rate_nm_min (Mono_Enum : integer; Scan_Rate : double): wordbool;
function ARC_Mono_Start_Scan_To_nm (Mono_Enum : integer; Wavelength_nm : double): wordbool;
function ARC_Mono_Scan_Done (Mono_Enum : integer; out Done_Moving : boolean;out Current_Wavelength_nm : double): wordbool;
function ARC_Mono_Start_Jog (Mono_Enum : integer; Jog_MaxRate : boolean; JogUp : boolean): wordbool;
function ARC_Mono_Stop_Jog (Mono_Enum : integer): wordbool;

// Serial Number Integers
function ARC_get_Filter_Serial_int32(Filter_Enum : integer) : integer;
function ARC_get_Mono_Serial_int32(Mono_Enum : integer) : integer;

// Pre Open Serial Number Integers
function ARC_get_Filter_preOpen_Serial_int32(Filter_Enum : integer; out Serial_int32 : integer) : wordbool;
function ARC_get_Mono_preOpen_Serial_int32(Mono_Enum : integer; out Serial_int32 : integer) : wordbool;

implementation
// define the Dll functions
type
 //Communications
 TARC_Search_For_Mono        = function (out Num_Found : integer): wordbool; stdcall;
 TARC_Open_Mono              = function (Enum_Num : integer; out Mono_Enum : integer): wordbool; stdcall;
 TARC_Open_Mono_Port         = function ( Com_Num : integer; out Mono_Enum : integer): wordbool; stdcall;
 TARC_Close_Mono             = function (Mono_Enum : integer): wordbool; stdcall;
 TARC_Valid_Mono_Enum        = function (Mono_Enum : integer): wordbool; stdcall;
 TARC_get_Mono_preOpen_Model = function (Enum_Num : integer; out Model_Var : olevariant) : wordbool; stdcall;
 TARC_get_Mono_preOpen_Model_int32
                             = function (Enum_Num : integer; out Model_int32 : integer) : wordbool; stdcall;
 TARC_Ver                    = function (out Major : integer; out Minor : integer; out Build : integer) : wordbool; stdcall;
 // direct communications function
 TARC_Send_CMD_To_Mono       = function (Mono_Enum : integer; SendCMD : olevariant; out RCV_Var : olevariant; ms_Timeout : integer) : wordbool; stdcall;

 //Information
 TARC_get_Mono_Model         = function (Mono_Enum : integer;out Model_Var : olevariant): wordbool; stdcall;
 TARC_get_Mono_Serial        = function (Mono_Enum : integer;out Serial_Var : olevariant): wordbool; stdcall;
 TARC_get_Mono_Focallength   = function (Mono_Enum : integer; out Focallength : double): wordbool; stdcall;
 TARC_get_Mono_HalfAngle     = function (Mono_Enum : integer; out HalfAngle : double): wordbool; stdcall;
 TARC_get_Mono_DetectorAngle = function (Mono_Enum : integer; out DetectorAngle : double): wordbool; stdcall;
 TARC_get_Mono_Double        = function (Mono_Enum : integer;out Double_Present : wordbool): wordbool; stdcall;
 TARC_get_Mono_Double_Subtractive
                             = function (Mono_Enum : integer;out Double_Subtractive : wordbool): wordbool; stdcall;
 TARC_get_Mono_Precision     = function (Mono_Enum : integer): integer; stdcall;

 // CCD Pixel Mapping
 TARC_get_Mono_Pixel_Map_nm    = function (Mono_Enum : integer; Pixel_Size_um : double; Number_X_Pixels : integer; Left_Adj_Percent : double; Center_Adj_Pixels : double; Right_Adj_Percent : double; Pixel_Map : pointer) : wordbool; stdcall;
 TARC_get_Mono_Pixel_Map_ang   = function (Mono_Enum : integer; Pixel_Size_um : double; Number_X_Pixels : integer; Left_Adj_Percent : double; Center_Adj_Pixels : double; Right_Adj_Percent : double; Pixel_Map : pointer) : wordbool; stdcall;
 TARC_get_Mono_Pixel_Map_eV    = function (Mono_Enum : integer; Pixel_Size_um : double; Number_X_Pixels : integer; Left_Adj_Percent : double; Center_Adj_Pixels : double; Right_Adj_Percent : double; Pixel_Map : pointer) : wordbool; stdcall;
 TARC_get_Mono_Pixel_Map_micron= function (Mono_Enum : integer; Pixel_Size_um : double; Number_X_Pixels : integer; Left_Adj_Percent : double; Center_Adj_Pixels : double; Right_Adj_Percent : double; Pixel_Map : pointer) : wordbool; stdcall;
 TARC_get_Mono_Pixel_Map_absCM = function (Mono_Enum : integer; Pixel_Size_um : double; Number_X_Pixels : integer; Left_Adj_Percent : double; Center_Adj_Pixels : double; Right_Adj_Percent : double; Pixel_Map : pointer) : wordbool; stdcall;
 TARC_get_Mono_Pixel_Map_relCM = function (Mono_Enum : integer; Pixel_Size_um : double; Number_X_Pixels : integer; Laser_Line_nm : double; Left_Adj_Percent : double; Center_Adj_Pixels : double; Right_Adj_Percent : double; Pixel_Map : pointer) : wordbool; stdcall;

 //Wavelength
 TARC_get_Mono_Wavelength_nm        = function (Mono_Enum : integer; out Wavelength : double): wordbool; stdcall;
 TARC_set_Mono_Wavelength_nm        = function (Mono_Enum : integer; Wavelength : double): wordbool; stdcall;
 TARC_get_Mono_Wavelength_ang       = function (Mono_Enum : integer; out Wavelength : double): wordbool; stdcall;
 TARC_set_Mono_Wavelength_ang       = function (Mono_Enum : integer; Wavelength : double): wordbool; stdcall;
 TARC_get_Mono_Wavelength_eV        = function (Mono_Enum : integer; out Wavelength : double): wordbool; stdcall;
 TARC_set_Mono_Wavelength_eV        = function (Mono_Enum : integer; Wavelength : double): wordbool; stdcall;
 TARC_get_Mono_Wavelength_micron    = function (Mono_Enum : integer; out Wavelength : double): wordbool; stdcall;
 TARC_set_Mono_Wavelength_micron    = function (Mono_Enum : integer; Wavelength : double): wordbool; stdcall;
 TARC_get_Mono_Wavelength_absCM     = function (Mono_Enum : integer; out Wavelength : double): wordbool; stdcall;
 TARC_set_Mono_Wavelength_absCM     = function (Mono_Enum : integer; Wavelength : double): wordbool; stdcall;
 TARC_get_Mono_Wavelength_relCM     = function (Mono_Enum : integer; Center_nm : double; out Wavelength : double): wordbool; stdcall;
 TARC_set_Mono_Wavelength_relCM     = function (Mono_Enum : integer; Center_nm : double; Wavelength : double): wordbool; stdcall;
 TARC_get_Mono_Wavelength_Cutoff_nm = function (Mono_Enum : integer; out Wavelength : double): wordbool; stdcall;
 TARC_get_Mono_Wavelength_Min_nm    = function (Mono_Enum : integer; out Wavelength : double): wordbool; stdcall;

 //Grating
 TARC_get_Mono_Turret            = function (Mono_Enum : integer; out Turret : integer): wordbool; stdcall;
 TARC_set_Mono_Turret            = function (Mono_Enum : integer; Turret : integer): wordbool; stdcall;
 TARC_get_Mono_Turret_Gratings   = function (Mono_Enum : integer; out Gratings_Per_Turret : integer): wordbool; stdcall;
 TARC_get_Mono_Turret_Max        = function (Mono_Enum : integer; out Max_Turret_Number : integer): wordbool; stdcall;
 TARC_get_Mono_Grating           = function (Mono_Enum : integer; out Grating : integer): wordbool; stdcall;
 TARC_set_Mono_Grating           = function (Mono_Enum : integer; Grating : integer): wordbool; stdcall;
 TARC_get_Mono_Grating_Blaze     = function (Mono_Enum : integer; Grating : integer; out Blaze_Var : olevariant): wordbool; stdcall;
 TARC_get_Mono_Grating_Density   = function (Mono_Enum : integer; Grating : integer; out Groove_MM : integer): wordbool; stdcall;
 TARC_get_Mono_Grating_Installed = function (Mono_Enum : integer; Grating : integer) : wordbool; stdcall;
 TARC_get_Mono_Grating_Max       = function (Mono_Enum : integer; out Max_Grating_Number : integer): wordbool; stdcall;

 //Mirrors
 TARC_get_Mono_Diverter_Valid   = function (Mono_Enum : integer; Diverter_Num : integer): wordbool; stdcall;
 TARC_get_Mono_Diverter_Pos     = function (Mono_Enum : integer; Diverter_Num : integer; out Diverter_Pos : integer): wordbool; stdcall;
 TARC_set_Mono_Diverter_Pos     = function (Mono_Enum : integer; Diverter_Num : integer; Diverter_Pos : integer): wordbool; stdcall;
 TARC_get_Mono_Diverter_Pos_Var = function (Mono_Enum : integer; Diverter_Num : integer; out Diverter_Pos : olevariant): wordbool; stdcall;

 //Slits
 TARC_get_Mono_Slit_Type     = function (Mono_Enum : integer; Slit_Pos : integer; out Slit_Type : integer): wordbool; stdcall;
 TARC_get_Mono_Slit_Type_Var = function (Mono_Enum : integer; Slit_Pos : integer; out Slit_Type : olevariant): wordbool; stdcall;
 TARC_get_Mono_Slit_Width    = function (Mono_Enum : integer; Slit_Pos : integer; out Slit_Width : integer): wordbool; stdcall;
 TARC_set_Mono_Slit_Width    = function (Mono_Enum : integer; Slit_Pos : integer; Slit_Width : integer): wordbool; stdcall;
 TARC_Mono_Slit_Home         = function (Mono_Enum : integer; Slit_Pos : integer): wordbool; stdcall;
 TARC_Mono_Slit_Name_Var     = function (Slit_Num : integer; out Slit_Name_Var : olevariant) : wordbool; stdcall;
 TARC_Calc_Mono_Slit_BandPass= function (Mono_Enum : integer; Slit_Pos : integer; SlitWidth : double; out BandPass : double): wordbool; stdcall;
 TARC_get_Mono_Slit_BandPass = function (Mono_Enum : integer; Slit_Pos : integer; out BandPass : double): wordbool; stdcall;
 TARC_set_Mono_Slit_BandPass = function (Mono_ENum : integer; Slit_Pos : integer; BandPass : double): wordbool; stdcall;
 // Special Double Function
 TARC_get_Mono_Double_Intermediate_Slit = function (Mono_Enum : integer; out Slit_Pos : integer): wordbool; stdcall;
 TARC_get_Mono_Exit_Slit     = function (Mono_Enum : integer; out Slit_Pos : integer): wordbool; stdcall;

 //Filter
 TARC_get_Mono_Filter_Present  = function (Mono_Enum : integer): wordbool; stdcall;
 TARC_get_Mono_Filter_Position = function (Mono_Enum : integer; out Position : integer): wordbool; stdcall;
 TARC_set_Mono_Filter_Position = function (Mono_Enum : integer; Position : integer): wordbool; stdcall;
 TARC_get_Mono_Filter_Min_Pos  = function (Mono_Enum : integer; out Position : integer): wordbool; stdcall;
 TARC_get_Mono_Filter_Max_Pos  = function (Mono_Enum : integer; out Position : integer): wordbool; stdcall;
 TARC_Mono_Filter_Home         = function (Mono_Enum : integer): wordbool; stdcall;

 // advanced functions
 // Gear
 TARC_get_Mono_Int_Led_On = function (Mono_Enum : integer): wordbool; stdcall;
 TARC_set_Mono_Int_Led    = function (Mono_Enum : integer; Led_State : wordbool): wordbool; stdcall;
 TARC_get_Mono_Motor_Int  = function (Mono_Enum : integer): wordbool; stdcall;
 TARC_get_Mono_Wheel_Int  = function (Mono_Enum : integer): wordbool; stdcall;
 TARC_Mono_Move_Steps     = function (Mono_Enum : integer; Num_Steps : integer): wordbool; stdcall;

 // Grating Values
 TARC_get_Mono_Init_Grating         = function (Mono_Enum : integer; out Init_Grating : integer): wordbool; stdcall;
 TARC_set_Mono_Init_Grating         = function (Mono_Enum : integer; Init_Grating : integer): wordbool; stdcall;
 TARC_get_Mono_Init_Wave_nm         = function (Mono_Enum : integer; out Init_Wave : double): wordbool; stdcall;
 TARC_set_Mono_Init_Wave_nm         = function (Mono_Enum : integer; Init_Wave : double): wordbool; stdcall;
 TARC_get_Mono_Init_ScanRate_nm     = function (Mono_Enum : integer; out Init_ScanRate : double): wordbool; stdcall;
 TARC_set_Mono_Init_ScanRate_nm     = function (Mono_Enum : integer;  Init_ScanRate : double): wordbool; stdcall;
 TARC_get_Mono_Grating_Offset       = function (Mono_Enum : integer; Grating : integer; out Offset : integer): wordbool; stdcall;
 TARC_get_Mono_Grating_Gadjust      = function (Mono_Enum : integer; Grating : integer; out Gadjust : integer): wordbool; stdcall;
 TARC_set_Mono_Grating_Offset       = function (Mono_Enum : integer; Grating : integer; Offset : integer): wordbool; stdcall;
 TARC_set_Mono_Grating_Gadjust      = function (Mono_Enum : integer; Grating : integer; Gadjust : integer): wordbool; stdcall;
 TARC_Mono_Grating_Calc_Offset      = function (Mono_Enum : integer; Grating : integer; Wave : double; RefWave : double; out newOffset : integer): wordbool; stdcall;
 TARC_Mono_Grating_Calc_Gadjust     = function (Mono_Enum : integer; Grating : integer; Wave : double; RefWave : double; out newGadjust : integer): wordbool; stdcall;
 TARC_Mono_Grating_UnInstall        = function (Mono_Enum : integer; Grating : integer): wordbool; stdcall;
 TARC_Mono_Grating_Install          = function (Mono_Enum : integer; Grating : integer; Density : Integer; Blaze : OleVariant; NMBlaze : boolean; HOLBlaze : boolean; MIRROR : boolean): wordbool; stdcall;
 TARC_Mono_Reset                    = function (Mono_Enum : integer): wordbool; stdcall;
 TARC_Mono_Restore_Factory_Settings = function (Mono_Enum : integer) : wordbool; stdcall;
 TARC_Mono_Save_Factory_Settings    = function (Mono_Enum : integer; password : double) : wordbool; stdcall;

 // Scan Values
 TARC_get_Mono_Scan_Rate_nm_min   = function (Mono_Enum : integer; out Scan_Rate : double): wordbool; stdcall;
 TARC_set_Mono_Scan_Rate_nm_min   = function (Mono_Enum : integer; Scan_Rate : double): wordbool; stdcall;
 TARC_Mono_Start_Scan_To_nm       = function (Mono_Enum : integer; Wavelength_nm : double): wordbool; stdcall;
 TARC_Mono_Scan_Done              = function (Mono_Enum : integer; out Done_Moving : boolean;out Current_Wavelength_nm : double): wordbool; stdcall;
 TARC_Mono_Start_Jog              = function (Mono_Enum : integer; Jog_MaxRate : boolean; JogUp : boolean): wordbool; stdcall;
 TARC_Mono_Stop_Jog               = function (Mono_Enum : integer): wordbool; stdcall;

 // Serial Number Integers
 TARC_get_Filter_Serial_int32     = function (Filter_Enum : integer) : integer; stdcall;
 TARC_get_Mono_Serial_int32       = function (Mono_Enum : integer) : integer; stdcall;

 // Pre Open Serial Number Integers
 TARC_get_Filter_preOpen_Serial_int32 = function (Filter_Enum : integer; out Serial_int32 : integer) : wordbool; stdcall;
 TARC_get_Mono_preOpen_Serial_int32   = function (ono_Enum : integer; out Serial_int32 : integer) : wordbool; stdcall;
// Local functions
var
 //Communications
 Local_ARC_Search_For_Mono        : TARC_Search_For_Mono;
 Local_ARC_Open_Mono              : TARC_Open_Mono;
 Local_ARC_Open_Mono_Port         : TARC_Open_Mono_Port;
 Local_ARC_Close_Mono             : TARC_Close_Mono;
 Local_ARC_Valid_Mono_Enum        : TARC_Valid_Mono_Enum;
 Local_ARC_get_Mono_preOpen_Model : TARC_get_Mono_preOpen_Model;
 Local_ARC_get_Mono_preOpen_Model_int32 : TARC_get_Mono_preOpen_Model_int32;
 Local_ARC_Ver                    : TARC_Ver;
 // direct communications function
 Local_ARC_Send_CMD_To_Mono       : TARC_Send_CMD_To_Mono;

 //Information
 Local_ARC_get_Mono_Model         : TARC_get_Mono_Model;
 Local_ARC_get_Mono_Serial        : TARC_get_Mono_Serial;
 Local_ARC_get_Mono_Focallength   : TARC_get_Mono_Focallength;
 Local_ARC_get_Mono_HalfAngle     : TARC_get_Mono_HalfAngle;
 Local_ARC_get_Mono_DetectorAngle : TARC_get_Mono_DetectorAngle;
 Local_ARC_get_Mono_Double        : TARC_get_Mono_Double;
 Local_ARC_get_Mono_Double_Subtractive
                                  : TARC_get_Mono_Double_Subtractive;
 Local_ARC_get_Mono_Precision     : TARC_get_Mono_Precision;

 // CCD Pixel Mapping
 Local_ARC_get_Mono_Pixel_Map_nm     : TARC_get_Mono_Pixel_Map_nm;
 Local_ARC_get_Mono_Pixel_Map_ang    : TARC_get_Mono_Pixel_Map_ang;
 Local_ARC_get_Mono_Pixel_Map_eV     : TARC_get_Mono_Pixel_Map_eV;
 Local_ARC_get_Mono_Pixel_Map_micron : TARC_get_Mono_Pixel_Map_micron;
 Local_ARC_get_Mono_Pixel_Map_absCM  : TARC_get_Mono_Pixel_Map_absCM;
 Local_ARC_get_Mono_Pixel_Map_relCM  : TARC_get_Mono_Pixel_Map_relCM;

 //Wavelength
 Local_ARC_get_Mono_Wavelength_nm        : TARC_get_Mono_Wavelength_nm;
 Local_ARC_set_Mono_Wavelength_nm        : TARC_set_Mono_Wavelength_nm;
 Local_ARC_get_Mono_Wavelength_ang       : TARC_get_Mono_Wavelength_ang;
 Local_ARC_set_Mono_Wavelength_ang       : TARC_set_Mono_Wavelength_ang;
 Local_ARC_get_Mono_Wavelength_eV        : TARC_get_Mono_Wavelength_eV;
 Local_ARC_set_Mono_Wavelength_eV        : TARC_set_Mono_Wavelength_eV;
 Local_ARC_get_Mono_Wavelength_micron    : TARC_get_Mono_Wavelength_micron;
 Local_ARC_set_Mono_Wavelength_micron    : TARC_set_Mono_Wavelength_micron;
 Local_ARC_get_Mono_Wavelength_absCM     : TARC_get_Mono_Wavelength_absCM;
 Local_ARC_set_Mono_Wavelength_absCM     : TARC_set_Mono_Wavelength_absCM;
 Local_ARC_get_Mono_Wavelength_relCM     : TARC_get_Mono_Wavelength_relCM;
 Local_ARC_set_Mono_Wavelength_relCM     : TARC_set_Mono_Wavelength_relCM;
 Local_ARC_get_Mono_Wavelength_Cutoff_nm : TARC_get_Mono_Wavelength_Cutoff_nm;
 Local_ARC_get_Mono_Wavelength_Min_nm    : TARC_get_Mono_Wavelength_Min_nm;

 //Grating
 Local_ARC_get_Mono_Turret            : TARC_get_Mono_Turret;
 Local_ARC_set_Mono_Turret            : TARC_set_Mono_Turret;
 Local_ARC_get_Mono_Turret_Gratings   : TARC_get_Mono_Turret_Gratings;
 Local_ARC_get_Mono_Turret_Max        : TARC_get_Mono_Turret_Max;
 Local_ARC_get_Mono_Grating           : TARC_get_Mono_Grating;
 Local_ARC_set_Mono_Grating           : TARC_set_Mono_Grating;
 Local_ARC_get_Mono_Grating_Blaze     : TARC_get_Mono_Grating_Blaze;
 Local_ARC_get_Mono_Grating_Density   : TARC_get_Mono_Grating_Density;
 Local_ARC_get_Mono_Grating_Installed : TARC_get_Mono_Grating_Installed;
 Local_ARC_get_Mono_Grating_Max       : TARC_get_Mono_Grating_Max;

 //Mirrors
 Local_ARC_get_Mono_Diverter_Valid   : TARC_get_Mono_Diverter_Valid;
 Local_ARC_get_Mono_Diverter_Pos     : TARC_get_Mono_Diverter_Pos;
 Local_ARC_set_Mono_Diverter_Pos     : TARC_set_Mono_Diverter_Pos;
 Local_ARC_get_Mono_Diverter_Pos_Var : TARC_get_Mono_Diverter_Pos_Var;

 //Slits
 Local_ARC_get_Mono_Slit_Type      : TARC_get_Mono_Slit_Type;
 Local_ARC_get_Mono_Slit_Type_Var  : TARC_get_Mono_Slit_Type_Var;
 Local_ARC_get_Mono_Slit_Width     : TARC_get_Mono_Slit_Width;
 Local_ARC_set_Mono_Slit_Width     : TARC_set_Mono_Slit_Width;
 Local_ARC_Mono_Slit_Home          : TARC_Mono_Slit_Home;
 Local_ARC_Mono_Slit_Name_Var      : TARC_Mono_Slit_Name_Var;
 Local_ARC_Calc_Mono_Slit_BandPass : TARC_Calc_Mono_Slit_BandPass;
 Local_ARC_get_Mono_Slit_BandPass  : TARC_get_Mono_Slit_BandPass;
 Local_ARC_set_Mono_Slit_BandPass  : TARC_set_Mono_Slit_BandPass;
 // Special Double Function
 Local_ARC_get_Mono_Double_Intermediate_Slit
                                   : TARC_get_Mono_Double_Intermediate_Slit;
 Local_ARC_get_Mono_Exit_Slit      : TARC_get_Mono_Exit_Slit;

 //Filter
 Local_ARC_get_Mono_Filter_Present  : TARC_get_Mono_Filter_Present;
 Local_ARC_get_Mono_Filter_Position : TARC_get_Mono_Filter_Position;
 Local_ARC_set_Mono_Filter_Position : TARC_set_Mono_Filter_Position;
 Local_ARC_get_Mono_Filter_Min_Pos  : TARC_get_Mono_Filter_Min_Pos;
 Local_ARC_get_Mono_Filter_Max_Pos  : TARC_get_Mono_Filter_Max_Pos;
 Local_ARC_Mono_Filter_Home         : TARC_Mono_Filter_Home;

 // advanced functions
 // Gear
 Local_ARC_get_Mono_Int_Led_On : TARC_get_Mono_Int_Led_On;
 Local_ARC_set_Mono_Int_Led    : TARC_set_Mono_Int_Led;
 Local_ARC_get_Mono_Motor_Int  : TARC_get_Mono_Motor_Int;
 Local_ARC_get_Mono_Wheel_Int  : TARC_get_Mono_Wheel_Int;
 Local_ARC_Mono_Move_Steps     : TARC_Mono_Move_Steps;

 // Grating Values
 Local_ARC_get_Mono_Init_Grating         : TARC_get_Mono_Init_Grating;
 Local_ARC_set_Mono_Init_Grating         : TARC_set_Mono_Init_Grating;
 Local_ARC_get_Mono_Init_Wave_nm         : TARC_get_Mono_Init_Wave_nm;
 Local_ARC_set_Mono_Init_Wave_nm         : TARC_set_Mono_Init_Wave_nm;
 Local_ARC_get_Mono_Init_ScanRate_nm     : TARC_get_Mono_Init_ScanRate_nm;
 Local_ARC_set_Mono_Init_ScanRate_nm     : TARC_set_Mono_Init_ScanRate_nm;
 Local_ARC_get_Mono_Grating_Offset       : TARC_get_Mono_Grating_Offset;
 Local_ARC_get_Mono_Grating_Gadjust      : TARC_get_Mono_Grating_Gadjust;
 Local_ARC_set_Mono_Grating_Offset       : TARC_set_Mono_Grating_Offset;
 Local_ARC_set_Mono_Grating_Gadjust      : TARC_set_Mono_Grating_Gadjust;
 Local_ARC_Mono_Grating_Calc_Offset      : TARC_Mono_Grating_Calc_Offset;
 Local_ARC_Mono_Grating_Calc_Gadjust     : TARC_Mono_Grating_Calc_Gadjust;
 Local_ARC_Mono_Grating_UnInstall        : TARC_Mono_Grating_UnInstall;
 Local_ARC_Mono_Grating_Install          : TARC_Mono_Grating_Install;
 Local_ARC_Mono_Reset                    : TARC_Mono_Reset;
 Local_ARC_Mono_Restore_Factory_Settings : TARC_Mono_Restore_Factory_Settings;
 Local_ARC_Mono_Save_Factory_Settings    : TARC_Mono_Save_Factory_Settings;

 // Scan Values
 Local_ARC_get_Mono_Scan_Rate_nm_min : TARC_get_Mono_Scan_Rate_nm_min;
 Local_ARC_set_Mono_Scan_Rate_nm_min : TARC_set_Mono_Scan_Rate_nm_min;
 Local_ARC_Mono_Start_Scan_To_nm     : TARC_Mono_Start_Scan_To_nm;
 Local_ARC_Mono_Scan_Done            : TARC_Mono_Scan_Done;
 Local_ARC_Mono_Start_Jog            : TARC_Mono_Start_Jog;
 Local_ARC_Mono_Stop_Jog             : TARC_Mono_Stop_Jog;

 // Serial Number Integers
 Local_ARC_get_Filter_Serial_int32 : TARC_get_Filter_Serial_int32;
 Local_ARC_get_Mono_Serial_int32   : TARC_get_Mono_Serial_int32;

 // Pre Open Serial Number Integers
 Local_ARC_get_Filter_preOpen_Serial_int32 : TARC_get_Filter_preOpen_Serial_int32;
 Local_ARC_get_Mono_preOpen_Serial_int32   : TARC_get_Mono_preOpen_Serial_int32;
 
// ARC_SpectraPro.dll encapsulation handle
ARC_SpectraPro : THandle;

// dynamic load and unload functions functions
function Dynamic_Load_ARC_SpectraPro_DLL : boolean;
begin
ARC_SpectraPro := 0;
try
ARC_SpectraPro := LoadLibrary ('ARC_SpectraPro.dll');
if ARC_SpectraPro <> 0
then begin
     //Communications
     @Local_ARC_Search_For_Mono := GetProcAddress (ARC_SpectraPro, 'ARC_Search_For_Mono');
     @Local_ARC_Open_Mono := GetProcAddress (ARC_SpectraPro, 'ARC_Open_Mono');
     @Local_ARC_Open_Mono_Port := GetProcAddress (ARC_SpectraPro, 'ARC_Open_Mono_Port');
     @Local_ARC_Close_Mono := GetProcAddress (ARC_SpectraPro, 'ARC_Close_Mono');
     @Local_ARC_Valid_Mono_Enum := GetProcAddress (ARC_SpectraPro, 'ARC_Valid_Mono_Enum');
     @Local_ARC_get_Mono_preOpen_Model := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_preOpen_Model');
     @Local_ARC_get_Mono_preOpen_Model_int32 := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_preOpen_Model_int32');
     @Local_ARC_Ver := GetProcAddress (ARC_SpectraPro, 'ARC_Ver');
     // direct communications function
     @Local_ARC_Send_CMD_To_Mono := GetProcAddress (ARC_SpectraPro, 'ARC_Send_CMD_To_Mono');

     //Information
     @Local_ARC_get_Mono_Model := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Model');
     @Local_ARC_get_Mono_Serial := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Serial');
     @Local_ARC_get_Mono_Focallength := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Focallength');
     @Local_ARC_get_Mono_HalfAngle := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_HalfAngle');
     @Local_ARC_get_Mono_DetectorAngle := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_DetectorAngle');
     @Local_ARC_get_Mono_Double := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Double');
     @Local_ARC_get_Mono_Double_Subtractive := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Double_Subtractive');
     @Local_ARC_get_Mono_Precision := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Precision');

      // CCD Pixel Mapping
     @Local_ARC_get_Mono_Pixel_Map_nm     := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Pixel_Map_nm');
     @Local_ARC_get_Mono_Pixel_Map_ang    := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Pixel_Map_ang');
     @Local_ARC_get_Mono_Pixel_Map_eV     := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Pixel_Map_eV');
     @Local_ARC_get_Mono_Pixel_Map_micron := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Pixel_Map_micron');
     @Local_ARC_get_Mono_Pixel_Map_absCM  := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Pixel_Map_absCM');
     @Local_ARC_get_Mono_Pixel_Map_relCM  := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Pixel_Map_relCM');

     //Wavelength
     @Local_ARC_get_Mono_Wavelength_nm := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Wavelength_nm');
     @Local_ARC_set_Mono_Wavelength_nm := GetProcAddress (ARC_SpectraPro, 'ARC_set_Mono_Wavelength_nm');
     @Local_ARC_get_Mono_Wavelength_ang := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Wavelength_ang');
     @Local_ARC_set_Mono_Wavelength_ang := GetProcAddress (ARC_SpectraPro, 'ARC_set_Mono_Wavelength_ang');
     @Local_ARC_get_Mono_Wavelength_eV := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Wavelength_eV');
     @Local_ARC_set_Mono_Wavelength_eV := GetProcAddress (ARC_SpectraPro, 'ARC_set_Mono_Wavelength_eV');
     @Local_ARC_get_Mono_Wavelength_micron := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Wavelength_micron');
     @Local_ARC_set_Mono_Wavelength_micron := GetProcAddress (ARC_SpectraPro, 'ARC_set_Mono_Wavelength_micron');
     @Local_ARC_get_Mono_Wavelength_absCM := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Wavelength_absCM');
     @Local_ARC_set_Mono_Wavelength_absCM := GetProcAddress (ARC_SpectraPro, 'ARC_set_Mono_Wavelength_absCM');
     @Local_ARC_get_Mono_Wavelength_relCM := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Wavelength_relCM');
     @Local_ARC_set_Mono_Wavelength_relCM := GetProcAddress (ARC_SpectraPro, 'ARC_set_Mono_Wavelength_relCM');
     @Local_ARC_get_Mono_Wavelength_Cutoff_nm := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Wavelength_Cutoff_nm');
     @Local_ARC_get_Mono_Wavelength_Min_nm := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Wavelength_Min_nm');

     //Grating
     @Local_ARC_get_Mono_Turret := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Turret');
     @Local_ARC_set_Mono_Turret := GetProcAddress (ARC_SpectraPro, 'ARC_set_Mono_Turret');
     @Local_ARC_get_Mono_Turret_Gratings := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Turret_Gratings');
     @Local_ARC_get_Mono_Turret_Max := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Turret_Max');
     @Local_ARC_get_Mono_Grating := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Grating');
     @Local_ARC_set_Mono_Grating := GetProcAddress (ARC_SpectraPro, 'ARC_set_Mono_Grating');
     @Local_ARC_get_Mono_Grating_Blaze := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Grating_Blaze');
     @Local_ARC_get_Mono_Grating_Density := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Grating_Density');
     @Local_ARC_get_Mono_Grating_Installed := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Grating_Installed');
     @Local_ARC_get_Mono_Grating_Max := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Grating_Max');

     //Mirrors
     @Local_ARC_get_Mono_Diverter_Valid := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Diverter_Valid');
     @Local_ARC_get_Mono_Diverter_Pos := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Diverter_Pos');
     @Local_ARC_set_Mono_Diverter_Pos := GetProcAddress (ARC_SpectraPro, 'ARC_set_Mono_Diverter_Pos');
     @Local_ARC_get_Mono_Diverter_Pos_Var := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Diverter_Pos_Var');

     //Slits
     @Local_ARC_get_Mono_Slit_Type := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Slit_Type');
     @Local_ARC_get_Mono_Slit_Type_Var := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Slit_Type_Var');
     @Local_ARC_get_Mono_Slit_Width := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Slit_Width');
     @Local_ARC_set_Mono_Slit_Width := GetProcAddress (ARC_SpectraPro, 'ARC_set_Mono_Slit_Width');
     @Local_ARC_Mono_Slit_Home := GetProcAddress (ARC_SpectraPro, 'ARC_Mono_Slit_Home');
     @Local_ARC_Mono_Slit_Name_Var := GetProcAddress (ARC_SpectraPro, 'ARC_Mono_Slit_Name_Var');
     @Local_ARC_Calc_Mono_Slit_BandPass := GetProcAddress (ARC_SpectraPro, 'ARC_Calc_Mono_Slit_BandPass');
     @Local_ARC_get_Mono_Slit_BandPass  := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Slit_BandPass');
     @Local_ARC_set_Mono_Slit_BandPass  := GetProcAddress (ARC_SpectraPro, 'ARC_set_Mono_Slit_BandPass');
     // Special Double Function
     @Local_ARC_get_Mono_Double_Intermediate_Slit := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Double_Intermediate_Slit');
     @Local_ARC_get_Mono_Exit_Slit                := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Exit_Slit');

     //Filter
     @Local_ARC_get_Mono_Filter_Present := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Filter_Present');
     @Local_ARC_get_Mono_Filter_Position := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Filter_Position');
     @Local_ARC_set_Mono_Filter_Position := GetProcAddress (ARC_SpectraPro, 'ARC_set_Mono_Filter_Position');
     @Local_ARC_get_Mono_Filter_Min_Pos := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Filter_Min_Pos');
     @Local_ARC_get_Mono_Filter_Max_Pos := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Filter_Max_Pos');
     @Local_ARC_Mono_Filter_Home := GetProcAddress (ARC_SpectraPro, 'ARC_Mono_Filter_Home');

     // advanced functions
     // Gear
     @Local_ARC_get_Mono_Int_Led_On := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Int_Led_On');
     @Local_ARC_set_Mono_Int_Led := GetProcAddress (ARC_SpectraPro, 'ARC_set_Mono_Int_Led');
     @Local_ARC_get_Mono_Motor_Int := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Motor_Int');
     @Local_ARC_get_Mono_Wheel_Int := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Wheel_Int');
     @Local_ARC_get_Mono_Wheel_Int := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Wheel_Int');
     @Local_ARC_Mono_Move_Steps := GetProcAddress (ARC_SpectraPro, 'ARC_Mono_Move_Steps');

     // Grating Values
     @Local_ARC_get_Mono_Init_Grating := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Init_Grating');
     @Local_ARC_set_Mono_Init_Grating := GetProcAddress (ARC_SpectraPro, 'ARC_set_Mono_Init_Grating');
     @Local_ARC_get_Mono_Init_Wave_nm := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Init_Wave_nm');
     @Local_ARC_set_Mono_Init_Wave_nm := GetProcAddress (ARC_SpectraPro, 'ARC_set_Mono_Init_Wave_nm');
     @Local_ARC_get_Mono_Init_ScanRate_nm := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Init_ScanRate_nm');
     @Local_ARC_set_Mono_Init_ScanRate_nm := GetProcAddress (ARC_SpectraPro, 'ARC_set_Mono_Init_ScanRate_nm');
     @Local_ARC_get_Mono_Grating_Offset := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Grating_Offset');
     @Local_ARC_get_Mono_Grating_Gadjust := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Grating_Gadjust');
     @Local_ARC_set_Mono_Grating_Offset := GetProcAddress (ARC_SpectraPro, 'ARC_set_Mono_Grating_Offset');
     @Local_ARC_set_Mono_Grating_Gadjust := GetProcAddress (ARC_SpectraPro, 'ARC_set_Mono_Grating_Gadjust');
     @Local_ARC_Mono_Grating_Calc_Offset := GetProcAddress (ARC_SpectraPro, 'ARC_Mono_Grating_Calc_Offset');
     @Local_ARC_Mono_Grating_Calc_Gadjust := GetProcAddress (ARC_SpectraPro, 'ARC_Mono_Grating_Calc_Gadjust');
     @Local_ARC_Mono_Grating_UnInstall := GetProcAddress (ARC_SpectraPro, 'ARC_Mono_Grating_UnInstall');
     @Local_ARC_Mono_Grating_Install := GetProcAddress (ARC_SpectraPro, 'ARC_Mono_Grating_Install');
     @Local_ARC_Mono_Reset := GetProcAddress (ARC_SpectraPro, 'ARC_Mono_Reset');
     @Local_ARC_Mono_Restore_Factory_Settings := GetProcAddress (ARC_SpectraPro, 'ARC_Mono_Restore_Factory_Settings');
     @Local_ARC_Mono_Save_Factory_Settings := GetProcAddress (ARC_SpectraPro, 'ARC_Mono_Save_Factory_Settings');

     // Scan Values
     @Local_ARC_get_Mono_Scan_Rate_nm_min := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Scan_Rate_nm_min');
     @Local_ARC_set_Mono_Scan_Rate_nm_min := GetProcAddress (ARC_SpectraPro, 'ARC_set_Mono_Scan_Rate_nm_min');
     @Local_ARC_Mono_Start_Scan_To_nm := GetProcAddress (ARC_SpectraPro, 'ARC_Mono_Start_Scan_To_nm');
     @Local_ARC_Mono_Scan_Done := GetProcAddress (ARC_SpectraPro, 'ARC_Mono_Scan_Done');
     @Local_ARC_Mono_Start_Jog := GetProcAddress (ARC_SpectraPro, 'ARC_Mono_Start_Jog');
     @Local_ARC_Mono_Stop_Jog := GetProcAddress (ARC_SpectraPro, 'ARC_Mono_Stop_Jog');

     // Serial Number Integers
     @Local_ARC_get_Filter_Serial_int32 := GetProcAddress (ARC_SpectraPro, 'ARC_get_Filter_Serial_int32');
     @Local_ARC_get_Mono_Serial_int32   := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_Serial_int32');

     // Pre Open Serial Number Integers
     @Local_ARC_get_Filter_preOpen_Serial_int32 := GetProcAddress (ARC_SpectraPro, 'ARC_get_Filter_preOpen_Serial_int32');
     @Local_ARC_get_Mono_preOpen_Serial_int32   := GetProcAddress (ARC_SpectraPro, 'ARC_get_Mono_preOpen_Serial_int32');

     // verify we properly linked the functions from the DLL
     if //Communications
        (@Local_ARC_Search_For_Mono <> nil)
        and
        (@Local_ARC_Open_Mono <> nil)
         and
        (@Local_ARC_Open_Mono_Port <> nil)
         and
        (@Local_ARC_Close_Mono <> nil)
         and
        (@Local_ARC_Valid_Mono_Enum <> nil)
         and
        (@Local_ARC_get_Mono_preOpen_Model <> nil)
         and
        (@Local_ARC_get_Mono_preOpen_Model_int32 <> nil)
         and
        (@Local_ARC_Ver <> nil)
         and
        // direct communications function
        (@Local_ARC_Send_CMD_To_Mono <> nil)
         and
        //Information
        (@Local_ARC_get_Mono_Model <> nil)
         and
        (@Local_ARC_get_Mono_Serial <> nil)
         and
        (@Local_ARC_get_Mono_Focallength <> nil)
         and
        (@Local_ARC_get_Mono_HalfAngle <> nil)
         and
        (@Local_ARC_get_Mono_DetectorAngle <> nil)
         and
        (@Local_ARC_get_Mono_Double <> nil)
         and
        (@Local_ARC_get_Mono_Double_Subtractive <> nil)
         and
        (@Local_ARC_get_Mono_Precision <> nil)
         and
        // CCD Pixel Mapping
        (@Local_ARC_get_Mono_Pixel_Map_nm <> nil)
         and
        (@Local_ARC_get_Mono_Pixel_Map_ang <> nil)
         and
        (@Local_ARC_get_Mono_Pixel_Map_eV <> nil)
         and
        (@Local_ARC_get_Mono_Pixel_Map_micron <> nil)
         and
        (@Local_ARC_get_Mono_Pixel_Map_absCM <> nil)
         and
        (@Local_ARC_get_Mono_Pixel_Map_relCM <> nil)
         and
        //Wavelength
        (@Local_ARC_get_Mono_Wavelength_nm <> nil)
         and
        (@Local_ARC_set_Mono_Wavelength_nm <> nil)
         and
        (@Local_ARC_get_Mono_Wavelength_ang <> nil)
         and
        (@Local_ARC_set_Mono_Wavelength_ang <> nil)
         and
        (@Local_ARC_get_Mono_Wavelength_eV <> nil)
         and
        (@Local_ARC_set_Mono_Wavelength_eV <> nil)
         and
        (@Local_ARC_get_Mono_Wavelength_micron <> nil)
         and
        (@Local_ARC_set_Mono_Wavelength_micron <> nil)
         and
        (@Local_ARC_get_Mono_Wavelength_absCM <> nil)
         and
        (@Local_ARC_set_Mono_Wavelength_absCM <> nil)
         and
        (@Local_ARC_get_Mono_Wavelength_relCM <> nil)
         and
        (@Local_ARC_set_Mono_Wavelength_relCM <> nil)
         and
        (@Local_ARC_get_Mono_Wavelength_Cutoff_nm <> nil)
         and
        (@Local_ARC_get_Mono_Wavelength_Min_nm <> nil)
         and
        //Grating
        (@Local_ARC_get_Mono_Turret <> nil)
         and
        (@Local_ARC_set_Mono_Turret <> nil)
         and
        (@Local_ARC_get_Mono_Turret_Gratings <> nil)
         and
        (@Local_ARC_get_Mono_Turret_Max <> nil)
         and
        (@Local_ARC_get_Mono_Grating <> nil)
         and
        (@Local_ARC_set_Mono_Grating <> nil)
         and
        (@Local_ARC_get_Mono_Grating_Blaze <> nil)
         and
        (@Local_ARC_get_Mono_Grating_Density <> nil)
         and
        (@Local_ARC_get_Mono_Grating_Installed <> nil)
         and
        (@Local_ARC_get_Mono_Grating_Max <> nil)
         and
        //Mirrors
        (@Local_ARC_get_Mono_Diverter_Valid <> nil)
         and
        (@Local_ARC_get_Mono_Diverter_Pos <> nil)
         and
        (@Local_ARC_set_Mono_Diverter_Pos <> nil)
         and
        (@Local_ARC_get_Mono_Diverter_Pos_Var <> nil)
         and
        //Slits
        (@Local_ARC_get_Mono_Slit_Type <> nil)
         and
        (@Local_ARC_get_Mono_Slit_Type_Var <> nil)
         and
        (@Local_ARC_get_Mono_Slit_Width <> nil)
         and
        (@Local_ARC_set_Mono_Slit_Width <> nil)
         and
        (@Local_ARC_Mono_Slit_Home <> nil)
         and
        (@Local_ARC_Mono_Slit_Name_Var <> nil)
         and
        (@Local_ARC_Calc_Mono_Slit_BandPass <> nil)
         and
        (@Local_ARC_get_Mono_Slit_BandPass <> nil)
         and
        (@Local_ARC_set_Mono_Slit_BandPass <> nil)
         and
        // Special Double Function
        (@Local_ARC_get_Mono_Double_Intermediate_Slit <> nil)
         and
        (@Local_ARC_get_Mono_Exit_Slit <> nil)
         and
        //Filter
        (@Local_ARC_get_Mono_Filter_Present <> nil)
         and
        (@Local_ARC_get_Mono_Filter_Position <> nil)
         and
        (@Local_ARC_set_Mono_Filter_Position <> nil)
         and
        (@Local_ARC_get_Mono_Filter_Min_Pos <> nil)
         and
        (@Local_ARC_get_Mono_Filter_Max_Pos <> nil)
         and
        (@Local_ARC_Mono_Filter_Home <> nil)
         and
        // advanced functions
        // Gear
        (@Local_ARC_get_Mono_Int_Led_On <> nil)
         and
        (@Local_ARC_set_Mono_Int_Led <> nil)
         and
        (@Local_ARC_get_Mono_Motor_Int <> nil)
         and
        (@Local_ARC_get_Mono_Wheel_Int <> nil)
         and
        (@Local_ARC_Mono_Move_Steps <> nil)
         and
        // Grating Values
        (@Local_ARC_get_Mono_Init_Grating <> nil)
         and
        (@Local_ARC_set_Mono_Init_Grating <> nil)
         and
        (@Local_ARC_get_Mono_Init_Wave_nm <> nil)
         and
        (@Local_ARC_set_Mono_Init_Wave_nm <> nil)
         and
        (@Local_ARC_get_Mono_Init_ScanRate_nm <> nil)
         and
        (@Local_ARC_set_Mono_Init_ScanRate_nm <> nil)
         and
        (@Local_ARC_get_Mono_Grating_Offset <> nil)
         and
        (@Local_ARC_get_Mono_Grating_Gadjust <> nil)
         and
        (@Local_ARC_set_Mono_Grating_Offset <> nil)
         and
        (@Local_ARC_set_Mono_Grating_Gadjust <> nil)
         and
        (@Local_ARC_Mono_Grating_Calc_Offset <> nil)
         and
        (@Local_ARC_Mono_Grating_Calc_Gadjust <> nil)
         and
        (@Local_ARC_Mono_Grating_UnInstall <> nil)
         and
        (@Local_ARC_Mono_Grating_Install <> nil)
         and
        (@Local_ARC_Mono_Reset <> nil)
         and
        (@Local_ARC_Mono_Restore_Factory_Settings <> nil)
         and
        (@Local_ARC_Mono_Save_Factory_Settings <> nil)
         and
        // Scan Values
        (@Local_ARC_get_Mono_Scan_Rate_nm_min <> nil)
         and
        (@Local_ARC_set_Mono_Scan_Rate_nm_min <> nil)
         and
        (@Local_ARC_Mono_Start_Scan_To_nm <> nil)
         and
        (@Local_ARC_Mono_Scan_Done <> nil)
         and
        (@Local_ARC_Mono_Start_Jog <> nil)
         and
        (@Local_ARC_Mono_Stop_Jog <> nil)
         and
        // Serial Number Integers
        (@Local_ARC_get_Filter_Serial_int32 <> nil)
         and
        (@Local_ARC_get_Mono_Serial_int32 <> nil)
         and
        // Pre Open Serial Number Integers
        (@Local_ARC_get_Filter_preOpen_Serial_int32 <> nil)
         and
        (@Local_ARC_get_Mono_preOpen_Serial_int32 <> nil)
     then begin
          // dynamically loaded ARC_SpectraPro.dll
          result := true;
          end
     else begin
          FreeLibrary ( ARC_SpectraPro );
          ARC_SpectraPro := 0;
          result := false;
          end;
     end
else begin
     result := false;
     end;
except
  result := false;
  end;
end;

procedure unload_ARC_SpectraPro_DLL;
//var
//CloseLoop : integer;
begin
// Delphi 6 unloads the library on exit and
// blows exeptions if you unload it for it
// so the unload library is commented out
{
if ARC_SpectraPro <> 0
then begin
     // force closed any open port
     for CloseLoop := 0 to 15
      do ARC_Close_Mono(CloseLoop);
     // unload the library
     FreeLibrary ( ARC_SpectraPro );
     ARC_SpectraPro := 0;
     end;
}
end;

//Communications
function ARC_Search_For_Mono(out Num_Found : integer): wordbool;
begin
result := Local_ARC_Search_For_Mono(Num_Found);
end;
function ARC_Open_Mono(Enum_Num : integer; out Mono_Enum : integer): wordbool;
begin
result := Local_ARC_Open_Mono(Enum_Num,Mono_Enum);
end;
function ARC_Open_Mono_Port(Com_Num : integer; out Mono_Enum : integer) : wordbool;
begin
result := Local_ARC_Open_Mono_Port(Com_Num,Mono_Enum);
end;
function ARC_Close_Mono(Mono_Enum : integer): wordbool;
begin
result := Local_ARC_Close_Mono(Mono_Enum);
end;
function ARC_Valid_Mono_Enum(Mono_Enum : integer): wordbool;
begin
result := Local_ARC_Valid_Mono_Enum(Mono_Enum);
end;
function ARC_get_Mono_preOpen_Model(Enum_Num : integer; out Model_str : string) : wordbool;
var
Model_Var : olevariant;
begin
result := Local_ARC_get_Mono_preOpen_Model(Enum_Num,Model_Var);
Model_Str := string(Model_Var);
end;
function ARC_get_Mono_preOpen_Model_int32(Enum_Num : integer; out Model_int32 : integer) : wordbool;
begin
result := Local_ARC_get_Mono_preOpen_Model_int32(Enum_Num,Model_int32);
end;
function ARC_Ver(out Major : integer; out Minor : integer; out Build : integer) : wordbool;
begin
result := Local_ARC_Ver(Major,Minor,Build);
end;
// direct communications function
function ARC_Send_CMD_To_Mono(Mono_Enum : integer; SendCMD : string; out RCV_str : string; ms_Timeout : integer) : wordbool;
var
Send_Var,RCV_Var : olevariant;
begin
Send_Var := SendCMD;
result := local_ARC_Send_CMD_To_Mono(Mono_Enum,Send_Var,RCV_Var,ms_Timeout);
RCV_Str := RCV_Var;
end;

//Information
function ARC_get_Mono_Model(Mono_Enum : integer;out Model_str : string): wordbool;
var
Model_var : olevariant;
begin
result := local_ARC_get_Mono_Model(Mono_Enum,Model_Var);
Model_Str := string(Model_Var);
end;
function ARC_get_Mono_Serial(Mono_Enum : integer;out Serial_str : string): wordbool;
var
Serial_var : olevariant;
begin
result := local_ARC_get_Mono_Serial(Mono_Enum,Serial_var);
Serial_str := string(Serial_var);
end;
function ARC_get_Mono_Focallength(Mono_Enum : integer; out Focallength : double): wordbool;
begin
result := local_ARC_get_Mono_Focallength(Mono_Enum,Focallength);
end;
function ARC_get_Mono_HalfAngle(Mono_Enum : integer; out HalfAngle : double): wordbool;
begin
result := local_ARC_get_Mono_HalfAngle(Mono_Enum,HalfAngle);
end;
function ARC_get_Mono_DetectorAngle(Mono_Enum : integer; out DetectorAngle : double): wordbool;
begin
result := local_ARC_get_Mono_DetectorAngle(Mono_Enum,DetectorAngle);
end;
function ARC_get_Mono_Double(Mono_Enum : integer;out Double_Present : wordbool): wordbool;
begin
result := local_ARC_get_Mono_Double(Mono_Enum,Double_Present);
end;
function ARC_get_Mono_Double_Subtractive(Mono_Enum : integer;out Double_Subtractive : wordbool): wordbool;
begin
result := Local_ARC_get_Mono_Double_Subtractive(Mono_Enum,Double_Subtractive);
end;
function ARC_get_Mono_Precision(Mono_Enum : integer): integer;
begin
result := local_ARC_get_Mono_Precision(Mono_Enum);
end;

// CCD Pixel Mapping
function ARC_get_Mono_Pixel_Map_nm    (Mono_Enum : integer; Pixel_Size_um : double; Number_X_Pixels : integer; Left_Adj_Percent : double; Center_Adj_Pixels : double; Right_Adj_Percent : double; Pixel_Map : pointer) : wordbool;
begin
result := Local_ARC_get_Mono_Pixel_Map_nm    (Mono_Enum,Pixel_Size_um,Number_X_Pixels,Left_Adj_Percent,Center_Adj_Pixels,Right_Adj_Percent,Pixel_Map);
end;
function ARC_get_Mono_Pixel_Map_ang   (Mono_Enum : integer; Pixel_Size_um : double; Number_X_Pixels : integer; Left_Adj_Percent : double; Center_Adj_Pixels : double; Right_Adj_Percent : double; Pixel_Map : pointer) : wordbool;
begin
result := Local_ARC_get_Mono_Pixel_Map_ang   (Mono_Enum,Pixel_Size_um,Number_X_Pixels,Left_Adj_Percent,Center_Adj_Pixels,Right_Adj_Percent,Pixel_Map);
end;
function ARC_get_Mono_Pixel_Map_eV    (Mono_Enum : integer; Pixel_Size_um : double; Number_X_Pixels : integer; Left_Adj_Percent : double; Center_Adj_Pixels : double; Right_Adj_Percent : double; Pixel_Map : pointer) : wordbool;
begin
result := Local_ARC_get_Mono_Pixel_Map_eV    (Mono_Enum,Pixel_Size_um,Number_X_Pixels,Left_Adj_Percent,Center_Adj_Pixels,Right_Adj_Percent,Pixel_Map);
end;
function ARC_get_Mono_Pixel_Map_micron(Mono_Enum : integer; Pixel_Size_um : double; Number_X_Pixels : integer; Left_Adj_Percent : double; Center_Adj_Pixels : double; Right_Adj_Percent : double; Pixel_Map : pointer) : wordbool;
begin
result := Local_ARC_get_Mono_Pixel_Map_micron(Mono_Enum,Pixel_Size_um,Number_X_Pixels,Left_Adj_Percent,Center_Adj_Pixels,Right_Adj_Percent,Pixel_Map);
end;
function ARC_get_Mono_Pixel_Map_absCM (Mono_Enum : integer; Pixel_Size_um : double; Number_X_Pixels : integer; Left_Adj_Percent : double; Center_Adj_Pixels : double; Right_Adj_Percent : double; Pixel_Map : pointer) : wordbool;
begin
result := Local_ARC_get_Mono_Pixel_Map_absCM (Mono_Enum,Pixel_Size_um,Number_X_Pixels,Left_Adj_Percent,Center_Adj_Pixels,Right_Adj_Percent,Pixel_Map);
end;
function ARC_get_Mono_Pixel_Map_relCM (Mono_Enum : integer; Pixel_Size_um : double; Number_X_Pixels : integer; Laser_Line_nm : double; Left_Adj_Percent : double; Center_Adj_Pixels : double; Right_Adj_Percent : double; Pixel_Map : pointer) : wordbool;
begin
result := Local_ARC_get_Mono_Pixel_Map_relCM (Mono_Enum,Pixel_Size_um,Number_X_Pixels,Laser_Line_nm,Left_Adj_Percent,Center_Adj_Pixels,Right_Adj_Percent,Pixel_Map);
end;

//Wavelength
function ARC_get_Mono_Wavelength_nm(Mono_Enum : integer; out Wavelength : double): wordbool; 
begin
result := local_ARC_get_Mono_Wavelength_nm(Mono_Enum,Wavelength);
end;
function ARC_set_Mono_Wavelength_nm(Mono_Enum : integer; Wavelength : double): wordbool;
begin
result := local_ARC_set_Mono_Wavelength_nm(Mono_Enum,Wavelength);
end;
function ARC_get_Mono_Wavelength_ang(Mono_Enum : integer; out Wavelength : double): wordbool;
begin
result := local_ARC_get_Mono_Wavelength_ang(Mono_Enum,Wavelength);
end;
function ARC_set_Mono_Wavelength_ang(Mono_Enum : integer; Wavelength : double): wordbool;
begin
result := local_ARC_set_Mono_Wavelength_ang(Mono_Enum,Wavelength);
end;
function ARC_get_Mono_Wavelength_eV(Mono_Enum : integer; out Wavelength : double): wordbool;
begin
result := local_ARC_get_Mono_Wavelength_eV(Mono_Enum,Wavelength);
end;
function ARC_set_Mono_Wavelength_eV(Mono_Enum : integer; Wavelength : double): wordbool;
begin
result := local_ARC_set_Mono_Wavelength_eV(Mono_Enum,Wavelength);
end;
function ARC_get_Mono_Wavelength_micron(Mono_Enum : integer; out Wavelength : double): wordbool;
begin
result := local_ARC_get_Mono_Wavelength_micron(Mono_Enum,Wavelength);
end;
function ARC_set_Mono_Wavelength_micron(Mono_Enum : integer; Wavelength : double): wordbool;
begin
result := local_ARC_set_Mono_Wavelength_micron(Mono_Enum,Wavelength);
end;
function ARC_get_Mono_Wavelength_absCM(Mono_Enum : integer; out Wavelength : double): wordbool;
begin
result := local_ARC_get_Mono_Wavelength_absCM(Mono_Enum,Wavelength);
end;
function ARC_set_Mono_Wavelength_absCM(Mono_Enum : integer; Wavelength : double): wordbool;
begin
result := local_ARC_set_Mono_Wavelength_absCM(Mono_Enum,Wavelength);
end;
function ARC_get_Mono_Wavelength_relCM(Mono_Enum : integer; Center_nm : double; out Wavelength : double): wordbool;
begin
result := local_ARC_get_Mono_Wavelength_relCM(Mono_Enum,Center_nm,Wavelength);
end;
function ARC_set_Mono_Wavelength_relCM(Mono_Enum : integer; Center_nm : double; Wavelength : double): wordbool;
begin
result := local_ARC_set_Mono_Wavelength_relCM(Mono_Enum,Center_nm,Wavelength);
end;
function ARC_get_Mono_Wavelength_Cutoff_nm(Mono_Enum : integer; out Wavelength : double): wordbool;
begin
result := local_ARC_get_Mono_Wavelength_Cutoff_nm(Mono_Enum,Wavelength);
end;
function ARC_get_Mono_Wavelength_Min_nm(Mono_Enum : integer; out Wavelength : double): wordbool;
begin
result := local_ARC_get_Mono_Wavelength_Min_nm(Mono_Enum,Wavelength);
end;

//Grating
function ARC_get_Mono_Turret(Mono_Enum : integer; out Turret : integer): wordbool;     
begin
result := local_ARC_get_Mono_Turret(Mono_Enum,Turret);
end;
function ARC_set_Mono_Turret(Mono_Enum : integer; Turret : integer): wordbool;
begin
result := local_ARC_set_Mono_Turret(Mono_Enum,Turret);
end;
function ARC_get_Mono_Turret_Gratings(Mono_Enum : integer; out Gratings_Per_Turret : integer): wordbool;
begin
result := local_ARC_get_Mono_Turret_Gratings(Mono_Enum,Gratings_Per_Turret);
end;
function ARC_get_Mono_Turret_Max(Mono_Enum : integer; out Max_Turret_Number : integer): wordbool;
begin
result := local_ARC_get_Mono_Turret_Max(Mono_Enum,Max_Turret_Number);
end;
function ARC_get_Mono_Grating(Mono_Enum : integer; out Grating : integer): wordbool;
begin
result := local_ARC_get_Mono_Grating(Mono_Enum,Grating);
end;
function ARC_set_Mono_Grating(Mono_Enum : integer; Grating : integer): wordbool;
begin
result := local_ARC_set_Mono_Grating(Mono_Enum,Grating);
end;
function ARC_get_Mono_Grating_Blaze(Mono_Enum : integer; Grating : integer; out Blaze_str : string): wordbool;
var
Blaze_var : olevariant;
begin
result := local_ARC_get_Mono_Grating_Blaze(Mono_Enum,Grating,Blaze_var);
Blaze_Str := string(Blaze_var);
end;
function ARC_get_Mono_Grating_Density(Mono_Enum : integer; Grating : integer; out Groove_MM : integer): wordbool; 
begin
result := local_ARC_get_Mono_Grating_Density(Mono_Enum,Grating,Groove_MM);
end;
function ARC_get_Mono_Grating_Installed(Mono_Enum : integer; Grating : integer) : wordbool;
begin
result := local_ARC_get_Mono_Grating_Installed(Mono_Enum,Grating);
end;
function ARC_get_Mono_Grating_Max(Mono_Enum : integer; out Max_Grating_Number : integer): wordbool;
begin
result := local_ARC_get_Mono_Grating_Max(Mono_Enum,Max_Grating_Number);
end;
//Mirrors
function ARC_get_Mono_Diverter_Valid(Mono_Enum : integer; Diverter_Num : integer): wordbool;
begin
result := local_ARC_get_Mono_Diverter_Valid(Mono_Enum,Diverter_Num);
end;
function ARC_get_Mono_Diverter_Pos(Mono_Enum : integer; Diverter_Num : integer; out Diverter_Pos : integer): wordbool;
begin
result := local_ARC_get_Mono_Diverter_Pos(Mono_Enum,Diverter_Num,Diverter_Pos);
end;
function ARC_set_Mono_Diverter_Pos(Mono_Enum : integer; Diverter_Num : integer; Diverter_Pos : integer): wordbool;
begin
result := local_ARC_set_Mono_Diverter_Pos(Mono_Enum,Diverter_Num,Diverter_Pos);
end;
function ARC_get_Mono_Diverter_Pos_str(Mono_Enum : integer; Diverter_Num : integer; out Diverter_Pos : string): wordbool;
var
Diverter_var : olevariant;
begin
result := local_ARC_get_Mono_Diverter_Pos_var(Mono_Enum,Diverter_Num,Diverter_var);
Diverter_Pos := string(Diverter_var);
end;

//Slits
function ARC_get_Mono_Slit_Type(Mono_Enum : integer; Slit_Pos : integer; out Slit_Type : integer): wordbool;
begin
result := local_ARC_get_Mono_Slit_Type(Mono_Enum,Slit_Pos,Slit_Type);
end;
function ARC_get_Mono_Slit_Type_str(Mono_Enum : integer; Slit_Pos : integer; out Slit_Type : string): wordbool;
var
Slit_var : olevariant;
begin
result := local_ARC_get_Mono_Slit_Type_var(Mono_Enum,Slit_Pos,Slit_var);
Slit_Type := string(Slit_var);
end;
function ARC_get_Mono_Slit_Width(Mono_Enum : integer; Slit_Pos : integer; out Slit_Width : integer): wordbool;
begin
result := local_ARC_get_Mono_Slit_Width(Mono_Enum,Slit_Pos,Slit_Width);
end;
function ARC_set_Mono_Slit_Width(Mono_Enum : integer; Slit_Pos : integer; Slit_Width : integer): wordbool;
begin
result := local_ARC_set_Mono_Slit_Width(Mono_Enum,Slit_Pos,Slit_Width);
end;
function ARC_Mono_Slit_Home(Mono_Enum : integer; Slit_Pos : integer): wordbool;                 
begin
result := local_ARC_Mono_Slit_Home(Mono_Enum,Slit_Pos);
end;
function ARC_Mono_Slit_Name_str(Slit_Num : integer; out Slit_Name_str : string) : wordbool;
var
Slit_Name_var : olevariant;
begin
result := local_ARC_Mono_Slit_Name_var(Slit_Num,Slit_Name_var);
Slit_Name_str := string(Slit_Name_var);
end;
function ARC_Calc_Mono_Slit_BandPass(Mono_Enum : integer; Slit_Pos : integer; SlitWidth : double; out BandPass : double): wordbool; stdcall;
begin
result := Local_ARC_Calc_Mono_Slit_BandPass(Mono_Enum,Slit_Pos,SlitWidth,BandPass);
end;
function ARC_get_Mono_Slit_BandPass(Mono_Enum : integer; Slit_Pos : integer; out BandPass : double): wordbool; stdcall;
begin
result := Local_ARC_get_Mono_Slit_BandPass(Mono_Enum,Slit_Pos,BandPass);
end;
function ARC_set_Mono_Slit_BandPass(Mono_ENum : integer; Slit_Pos : integer; BandPass : double): wordbool; stdcall;
begin
result := Local_ARC_set_Mono_Slit_BandPass(Mono_ENum,Slit_Pos,BandPass);
end;
// Special Double Function
function ARC_get_Mono_Double_Intermediate_Slit(Mono_Enum : integer; out Slit_Pos : integer): wordbool;
begin
result := Local_ARC_get_Mono_Double_Intermediate_Slit(Mono_Enum,Slit_Pos);
end;
function ARC_get_Mono_Exit_Slit(Mono_Enum : integer; out Slit_Pos : integer): wordbool;
begin
result := Local_ARC_get_Mono_Exit_Slit(Mono_Enum,Slit_Pos);
end;

//Filter
function ARC_get_Mono_Filter_Present(Mono_Enum : integer): wordbool;
begin
result := local_ARC_get_Mono_Filter_Present(Mono_Enum);
end;
function ARC_get_Mono_Filter_Position(Mono_Enum : integer; out Position : integer): wordbool;
begin
result := local_ARC_get_Mono_Filter_Position(Mono_Enum,Position);
end;
function ARC_set_Mono_Filter_Position(Mono_Enum : integer; Position : integer): wordbool;
begin
result := local_ARC_set_Mono_Filter_Position(Mono_Enum,Position);
end;
function ARC_get_Mono_Filter_Min_Pos(Mono_Enum : integer; out Position : integer): wordbool;
begin
result := local_ARC_get_Mono_Filter_Min_Pos(Mono_Enum,Position);
end;
function ARC_get_Mono_Filter_Max_Pos(Mono_Enum : integer; out Position : integer): wordbool;
begin
result := local_ARC_get_Mono_Filter_Max_Pos(Mono_Enum,Position);
end;
function ARC_Mono_Filter_Home(Mono_Enum : integer): wordbool;  
begin
result := local_ARC_Mono_Filter_Home(Mono_Enum);
end;

// advanced functions
// Gear
function ARC_get_Mono_Int_Led_On(Mono_Enum : integer): wordbool;
begin
result := local_ARC_get_Mono_Int_Led_On(Mono_Enum);
end;
function ARC_set_Mono_Int_Led(Mono_Enum : integer; Led_State : wordbool): wordbool;
begin
result := local_ARC_set_Mono_Int_Led(Mono_Enum,Led_State);
end;
function ARC_get_Mono_Motor_Int(Mono_Enum : integer): wordbool;
begin
result := local_ARC_get_Mono_Motor_Int(Mono_Enum);
end;
function ARC_get_Mono_Wheel_Int(Mono_Enum : integer): wordbool;
begin
result := local_ARC_get_Mono_Wheel_Int(Mono_Enum);
end;
function ARC_Mono_Move_Steps(Mono_Enum : integer; Num_Steps : integer): wordbool;
begin
result := local_ARC_Mono_Move_Steps(Mono_Enum,Num_Steps);
end;

// Grating Values
function ARC_get_Mono_Init_Grating(Mono_Enum : integer; out Init_Grating : integer): wordbool;
begin
result := local_ARC_get_Mono_Init_Grating(Mono_Enum,Init_Grating);
end;
function ARC_set_Mono_Init_Grating(Mono_Enum : integer; Init_Grating : integer): wordbool;
begin
result := local_ARC_set_Mono_Init_Grating(Mono_Enum,Init_Grating);
end;
function ARC_get_Mono_Init_Wave_nm(Mono_Enum : integer; out Init_Wave : double): wordbool;
begin
result := local_ARC_get_Mono_Init_Wave_nm(Mono_Enum,Init_Wave);
end;
function ARC_set_Mono_Init_Wave_nm(Mono_Enum : integer; Init_Wave : double): wordbool;
begin
result := local_ARC_set_Mono_Init_Wave_nm(Mono_Enum,Init_Wave);
end;
function ARC_get_Mono_Init_ScanRate_nm(Mono_Enum : integer; out Init_ScanRate : double): wordbool;
begin
result := local_ARC_get_Mono_Init_ScanRate_nm(Mono_Enum,Init_ScanRate);
end;
function ARC_set_Mono_Init_ScanRate_nm(Mono_Enum : integer;  Init_ScanRate : double): wordbool;
begin
result := local_ARC_set_Mono_Init_ScanRate_nm(Mono_Enum,Init_ScanRate);
end;
function ARC_get_Mono_Grating_Offset(Mono_Enum : integer; Grating : integer; out Offset : integer): wordbool;
begin
result := local_ARC_get_Mono_Grating_Offset(Mono_Enum,Grating,Offset);
end;
function ARC_get_Mono_Grating_Gadjust(Mono_Enum : integer; Grating : integer; out Gadjust : integer): wordbool;
begin
result := local_ARC_get_Mono_Grating_Gadjust(Mono_Enum,Grating,Gadjust);
end;
function ARC_set_Mono_Grating_Offset(Mono_Enum : integer; Grating : integer; Offset : integer): wordbool;
begin
result := local_ARC_set_Mono_Grating_Offset(Mono_Enum,Grating,Offset);
end;
function ARC_set_Mono_Grating_Gadjust(Mono_Enum : integer; Grating : integer; Gadjust : integer): wordbool;
begin
result := local_ARC_set_Mono_Grating_Gadjust(Mono_Enum,Grating,Gadjust);
end;
function ARC_Mono_Grating_Calc_Offset(Mono_Enum : integer; Grating : integer; Wave : double; RefWave : double; out newOffset : integer): wordbool;
begin
result := local_ARC_Mono_Grating_Calc_Offset(Mono_Enum,Grating,Wave,RefWave,newOffset);
end;
function ARC_Mono_Grating_Calc_Gadjust(Mono_Enum : integer; Grating : integer; Wave : double; RefWave : double; out newGadjust : integer): wordbool;
begin
result := local_ARC_Mono_Grating_Calc_Gadjust(Mono_Enum,Grating,Wave,RefWave,newGadjust);
end;
function ARC_Mono_Grating_UnInstall(Mono_Enum : integer; Grating : integer): wordbool;
begin
result := local_ARC_Mono_Grating_UnInstall(Mono_Enum,Grating);
end;
function ARC_Mono_Grating_Install(Mono_Enum : integer; Grating : integer; Density : Integer; Blaze : OleVariant; NMBlaze : boolean; HOLBlaze : boolean; MIRROR : boolean): wordbool;
begin
result := local_ARC_Mono_Grating_Install(Mono_Enum,Grating,Density,Blaze,NMBlaze,HOLBlaze,MIRROR);
end;
function ARC_Mono_Reset(Mono_Enum : integer): wordbool;
begin
result := local_ARC_Mono_Reset(Mono_Enum);
end;
function ARC_Mono_Restore_Factory_Settings(Mono_Enum : integer) : wordbool;
begin
result := local_ARC_Mono_Restore_Factory_Settings(Mono_Enum);
end;
function ARC_Mono_Save_Factory_Settings(Mono_Enum : integer; password : double) : wordbool;
begin
result := local_ARC_Mono_Save_Factory_Settings(Mono_Enum,password);
end;

// Scan Values
function ARC_get_Mono_Scan_Rate_nm_min (Mono_Enum : integer; out Scan_Rate : double): wordbool;
begin
result := local_ARC_get_Mono_Scan_Rate_nm_min (Mono_Enum,Scan_Rate);
end;
function ARC_set_Mono_Scan_Rate_nm_min (Mono_Enum : integer; Scan_Rate : double): wordbool;
begin
result := local_ARC_set_Mono_Scan_Rate_nm_min (Mono_Enum,Scan_Rate);
end;
function ARC_Mono_Start_Scan_To_nm (Mono_Enum : integer; Wavelength_nm : double): wordbool;
begin
result := local_ARC_Mono_Start_Scan_To_nm (Mono_Enum,Wavelength_nm);
end;
function ARC_Mono_Scan_Done (Mono_Enum : integer; out Done_Moving : boolean;out Current_Wavelength_nm : double): wordbool;
begin
result := local_ARC_Mono_Scan_Done (Mono_Enum,Done_Moving,Current_Wavelength_nm);
end;
function ARC_Mono_Start_Jog (Mono_Enum : integer; Jog_MaxRate : boolean; JogUp : boolean): wordbool;
begin
result := local_ARC_Mono_Start_Jog (Mono_Enum,Jog_MaxRate,JogUp);
end;
function ARC_Mono_Stop_Jog (Mono_Enum : integer): wordbool;
begin
result := local_ARC_Mono_Stop_Jog (Mono_Enum);
end;

// Serial Number Integers
function ARC_get_Filter_Serial_int32(Filter_Enum : integer) : integer;
begin
result := Local_ARC_get_Filter_Serial_int32(Filter_Enum);
end;
function ARC_get_Mono_Serial_int32(Mono_Enum : integer) : integer;
begin
result := Local_ARC_get_Mono_Serial_int32(Mono_Enum);
end;

// Pre Open Serial Number Integers
function ARC_get_Filter_preOpen_Serial_int32(Filter_Enum : integer; out Serial_int32 : integer) : wordbool;
begin
result := Local_ARC_get_Filter_preOpen_Serial_int32(Filter_Enum,Serial_int32);
end;
function ARC_get_Mono_preOpen_Serial_int32(Mono_Enum : integer; out Serial_int32 : integer) : wordbool;
begin
result := Local_ARC_get_Mono_preOpen_Serial_int32(Mono_Enum,Serial_int32);
end;

end.
