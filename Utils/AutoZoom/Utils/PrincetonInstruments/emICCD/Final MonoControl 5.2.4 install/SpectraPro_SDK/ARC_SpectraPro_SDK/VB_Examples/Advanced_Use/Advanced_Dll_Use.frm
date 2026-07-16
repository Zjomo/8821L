VERSION 5.00
Begin VB.Form Form1 
   Caption         =   "Form1"
   ClientHeight    =   5760
   ClientLeft      =   60
   ClientTop       =   345
   ClientWidth     =   12600
   LinkTopic       =   "Form1"
   ScaleHeight     =   5760
   ScaleWidth      =   12600
   StartUpPosition =   3  'Windows Default
   Begin VB.CommandButton SetGAdjBtn 
      Caption         =   "Set GAdj"
      Height          =   375
      Left            =   11520
      TabIndex        =   47
      Top             =   5280
      Width           =   975
   End
   Begin VB.CommandButton SetOffsetBtn 
      Caption         =   "Set Offset"
      Height          =   375
      Left            =   11520
      TabIndex        =   46
      Top             =   4560
      Width           =   975
   End
   Begin VB.CommandButton CalcGAdjBtn 
      Caption         =   "Calc GAdj"
      Height          =   375
      Left            =   9240
      TabIndex        =   45
      Top             =   5280
      Width           =   975
   End
   Begin VB.CommandButton CalcOffsetBtn 
      Caption         =   "Calc Offset"
      Height          =   375
      Left            =   9240
      TabIndex        =   44
      Top             =   4560
      Width           =   975
   End
   Begin VB.TextBox RefWaveEdt 
      Height          =   375
      Left            =   8280
      TabIndex        =   43
      Top             =   5280
      Width           =   855
   End
   Begin VB.TextBox DisWaveEdt 
      Height          =   375
      Left            =   8280
      TabIndex        =   42
      Top             =   4560
      Width           =   855
   End
   Begin VB.CommandButton UnInstallBtn 
      Caption         =   "UnInstall Grating"
      Height          =   375
      Left            =   10440
      TabIndex        =   38
      Top             =   3840
      Width           =   2055
   End
   Begin VB.CommandButton InstallGratBtn 
      Caption         =   "Install Grating"
      Height          =   375
      Left            =   8280
      TabIndex        =   37
      Top             =   3840
      Width           =   2055
   End
   Begin VB.TextBox BlazeEdt 
      Height          =   375
      Left            =   9840
      TabIndex        =   36
      Top             =   3360
      Width           =   855
   End
   Begin VB.TextBox GrooveEdt 
      Height          =   375
      Left            =   9840
      TabIndex        =   35
      Top             =   2880
      Width           =   855
   End
   Begin VB.TextBox GratEdt 
      Height          =   375
      Left            =   9840
      TabIndex        =   34
      Top             =   2400
      Width           =   375
   End
   Begin VB.Frame BlazeFrame 
      Height          =   1335
      Left            =   11040
      TabIndex        =   29
      Top             =   2400
      Width           =   1335
      Begin VB.OptionButton HOLRadio 
         Caption         =   "HOL Blaze"
         Height          =   255
         Left            =   120
         TabIndex        =   33
         Top             =   960
         Width           =   1095
      End
      Begin VB.OptionButton umRadio 
         Caption         =   "um Blaze"
         Height          =   255
         Left            =   120
         TabIndex        =   32
         Top             =   720
         Width           =   1095
      End
      Begin VB.OptionButton nmRadio 
         Caption         =   "nm Blaze"
         Height          =   255
         Left            =   120
         TabIndex        =   31
         Top             =   480
         Width           =   1095
      End
      Begin VB.OptionButton MirrorRadio 
         Caption         =   "Mirror"
         Height          =   255
         Left            =   120
         TabIndex        =   30
         Top             =   240
         Width           =   1095
      End
   End
   Begin VB.TextBox StopScanEdt 
      Height          =   375
      Left            =   9480
      TabIndex        =   24
      Top             =   1800
      Width           =   855
   End
   Begin VB.TextBox StartScanEdt 
      Height          =   375
      Left            =   9480
      TabIndex        =   23
      Top             =   1320
      Width           =   855
   End
   Begin VB.TextBox ScanRateEdt 
      Height          =   375
      Left            =   9480
      TabIndex        =   22
      Top             =   840
      Width           =   855
   End
   Begin VB.CommandButton ScanBtn 
      Caption         =   "Start Scan"
      Height          =   375
      Left            =   10440
      TabIndex        =   21
      Top             =   1800
      Width           =   1215
   End
   Begin VB.CommandButton SetRateBtn 
      Caption         =   "Set Scan Rate"
      Height          =   375
      Left            =   10440
      TabIndex        =   20
      Top             =   840
      Width           =   1215
   End
   Begin VB.CommandButton MoveStepsBtn 
      Caption         =   "Move Steps"
      Height          =   375
      Left            =   10440
      TabIndex        =   19
      Top             =   360
      Width           =   1215
   End
   Begin VB.TextBox StepsEdt 
      Height          =   375
      Left            =   9480
      TabIndex        =   18
      Top             =   360
      Width           =   855
   End
   Begin VB.CheckBox IntLedChk 
      Caption         =   "Int Led State"
      Height          =   255
      Left            =   8280
      TabIndex        =   16
      Top             =   120
      Width           =   1935
   End
   Begin VB.CommandButton ResetMonoBtn 
      Caption         =   "Reset Mono"
      Height          =   375
      Left            =   1080
      TabIndex        =   15
      Top             =   4200
      Width           =   2055
   End
   Begin VB.CommandButton ResetFactoryBtn 
      Caption         =   "Reset Factory"
      Height          =   375
      Left            =   1080
      TabIndex        =   14
      Top             =   3720
      Width           =   2055
   End
   Begin VB.CommandButton SetInitScanBtn 
      Caption         =   "Set Init Scan Rate"
      Height          =   375
      Left            =   1680
      TabIndex        =   13
      Top             =   3240
      Width           =   2055
   End
   Begin VB.CommandButton SetInitWaveBtn 
      Caption         =   "Set Init Wavelength"
      Height          =   375
      Left            =   1680
      TabIndex        =   12
      Top             =   2760
      Width           =   2055
   End
   Begin VB.CommandButton SetInitGratBtn 
      Caption         =   "Set Init Grating"
      Height          =   375
      Left            =   1680
      TabIndex        =   11
      Top             =   2280
      Width           =   2055
   End
   Begin VB.TextBox InitScanEdt 
      Height          =   375
      Left            =   720
      TabIndex        =   10
      Top             =   3240
      Width           =   855
   End
   Begin VB.TextBox InitWaveEdt 
      Height          =   375
      Left            =   720
      TabIndex        =   9
      Top             =   2760
      Width           =   855
   End
   Begin VB.TextBox InitGratEdt 
      Height          =   375
      Left            =   720
      TabIndex        =   8
      Top             =   2280
      Width           =   855
   End
   Begin VB.CommandButton MonoInfoBtn 
      Caption         =   "Display Mono State"
      Height          =   375
      Left            =   720
      TabIndex        =   7
      Top             =   1800
      Width           =   1815
   End
   Begin VB.CheckBox OpenChk 
      Caption         =   "Mono Open"
      Height          =   375
      Left            =   2280
      TabIndex        =   5
      Top             =   840
      Width           =   1575
   End
   Begin VB.CommandButton FindMonoBtn 
      Caption         =   "Find Mono(s)"
      Height          =   375
      Left            =   0
      TabIndex        =   4
      Top             =   360
      Width           =   1215
   End
   Begin VB.TextBox EnumEdt 
      Height          =   375
      Left            =   0
      TabIndex        =   3
      Text            =   "Enum "
      Top             =   840
      Width           =   855
   End
   Begin VB.CommandButton OpenBtn 
      Caption         =   "Open Enum"
      Height          =   375
      Left            =   960
      TabIndex        =   2
      Top             =   840
      Width           =   1215
   End
   Begin VB.CommandButton CloseBtn 
      Caption         =   "Close Enum"
      Height          =   375
      Left            =   960
      TabIndex        =   1
      Top             =   1320
      Width           =   1215
   End
   Begin VB.ListBox List1 
      Height          =   5715
      Left            =   3840
      TabIndex        =   0
      Top             =   0
      Width           =   4335
   End
   Begin VB.Label RefWaveLabel 
      Caption         =   "Ref Wave"
      Height          =   255
      Left            =   8280
      TabIndex        =   51
      Top             =   5040
      Width           =   1215
   End
   Begin VB.Label DisWavLabel 
      Caption         =   "Display Wave"
      Height          =   255
      Left            =   8280
      TabIndex        =   50
      Top             =   4320
      Width           =   1215
   End
   Begin VB.Label GAdjLabel 
      Caption         =   "."
      Height          =   255
      Left            =   10320
      TabIndex        =   49
      Top             =   5280
      Width           =   1215
   End
   Begin VB.Label OffsetLabel 
      Caption         =   "."
      Height          =   255
      Left            =   10320
      TabIndex        =   48
      Top             =   4560
      Width           =   1215
   End
   Begin VB.Label BlazeLabel 
      Caption         =   "Blaze String"
      Height          =   255
      Left            =   8280
      TabIndex        =   41
      Top             =   3360
      Width           =   1455
   End
   Begin VB.Label DensityLabel 
      Caption         =   "Groove Density"
      Height          =   255
      Left            =   8280
      TabIndex        =   40
      Top             =   2880
      Width           =   1455
   End
   Begin VB.Label GratNumLabel 
      Caption         =   "Grating Num"
      Height          =   255
      Left            =   8280
      TabIndex        =   39
      Top             =   2400
      Width           =   1455
   End
   Begin VB.Label ScannmLabel 
      Caption         =   "Scan : nm"
      Height          =   255
      Left            =   10440
      TabIndex        =   28
      Top             =   1440
      Width           =   1215
   End
   Begin VB.Label StopLabel 
      Caption         =   "Stop Scan nm"
      Height          =   255
      Left            =   8280
      TabIndex        =   27
      Top             =   1800
      Width           =   1215
   End
   Begin VB.Label StartLabel 
      Caption         =   "Start Scan nm"
      Height          =   255
      Left            =   8280
      TabIndex        =   26
      Top             =   1320
      Width           =   1215
   End
   Begin VB.Label RateLabel 
      Caption         =   "New Scan Rate"
      Height          =   255
      Left            =   8280
      TabIndex        =   25
      Top             =   840
      Width           =   1215
   End
   Begin VB.Label StepsLabel 
      Caption         =   "Steps To Move"
      Height          =   255
      Left            =   8280
      TabIndex        =   17
      Top             =   480
      Width           =   1215
   End
   Begin VB.Label DLLLabel 
      Caption         =   "ARC_SpectraPro.dll ver. "
      Height          =   255
      Left            =   0
      TabIndex        =   6
      Top             =   0
      Width           =   2895
   End
End
Attribute VB_Name = "Form1"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = False
'ARC_SpectraPro.dll functions
'High level Communications
Private Declare Function ARC_Search_For_Mono Lib "ARC_SpectraPro.dll" (Num_Found As Long) As Integer
Private Declare Function ARC_Open_Mono Lib "ARC_SpectraPro.dll" (ByVal Enum_Num As Long, Mono_Enum As Long) As Integer
Private Declare Function ARC_Close_Mono Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long) As Integer
Private Declare Function ARC_Valid_Mono_Enum Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long) As Integer
Private Declare Function ARC_get_Mono_preOpen_Model Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Model_Var As Variant) As Integer
Private Declare Function ARC_Ver Lib "ARC_SpectraPro.dll" (Majorval As Long, Minorval As Long, Buildval As Long) As Integer
'Direct Communications Function
Private Declare Function ARC_Send_CMD_To_Mono Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal SendCMD As Variant, RCV_Var As Variant, ByVal ms_Timeout As Long) As Integer

'Monochromator Information
Private Declare Function ARC_get_Mono_Model Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Model_str As Variant) As Integer
Private Declare Function ARC_get_Mono_Serial Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Serial_str As Variant) As Integer
Private Declare Function ARC_get_Mono_Focallength Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Focallength As Double) As Integer
Private Declare Function ARC_get_Mono_HalfAngle Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, HalfAngle As Double) As Integer
Private Declare Function ARC_get_Mono_DetectorAngle Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, DetectorAngle As Double) As Integer
Private Declare Function ARC_get_Mono_Double Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Double_Present As Integer) As Integer
Private Declare Function ARC_get_Mono_Precision Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long) As Long

'Monochromator Wavelength
Private Declare Function ARC_get_Mono_Wavelength_nm Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Wavelength As Double) As Integer
Private Declare Function ARC_set_Mono_Wavelength_nm Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Wavelength As Double) As Integer
Private Declare Function ARC_get_Mono_Wavelength_ang Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Wavelength As Double) As Integer
Private Declare Function ARC_set_Mono_Wavelength_ang Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Wavelength As Double) As Integer
Private Declare Function ARC_get_Mono_Wavelength_eV Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Wavelength As Double) As Integer
Private Declare Function ARC_set_Mono_Wavelength_eV Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Wavelength As Double) As Integer
Private Declare Function ARC_get_Mono_Wavelength_micron Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Wavelength As Double) As Integer
Private Declare Function ARC_set_Mono_Wavelength_micron Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Wavelength As Double) As Integer
Private Declare Function ARC_get_Mono_Wavelength_absCM Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Wavelength As Double) As Integer
Private Declare Function ARC_set_Mono_Wavelength_absCM Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Wavelength As Double) As Integer
Private Declare Function ARC_get_Mono_Wavelength_relCM Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Center_nm As Double, Wavelength As Double) As Integer
Private Declare Function ARC_set_Mono_Wavelength_relCM Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Center_nm As Double, ByVal Wavelength As Double) As Integer
Private Declare Function ARC_get_Mono_Wavelength_Cutoff_nm Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Wavelength As Double) As Integer
Private Declare Function ARC_get_Mono_Wavelength_Min_nm Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Wavelength As Double) As Integer

'Monochromator Grating
Private Declare Function ARC_get_Mono_Turret Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Turret As Long) As Integer
Private Declare Function ARC_set_Mono_Turret Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Turret As Long) As Integer
Private Declare Function ARC_get_Mono_Turret_Gratings Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Gratings_Per_Turret As Long) As Integer
Private Declare Function ARC_get_Mono_Grating Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Grating As Long) As Integer
Private Declare Function ARC_set_Mono_Grating Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Grating As Long) As Integer
Private Declare Function ARC_get_Mono_Grating_Blaze Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Grating As Long, Blaze_Var As Variant) As Integer
Private Declare Function ARC_get_Mono_Grating_Density Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Grating As Long, Groove_MM As Long) As Integer
Private Declare Function ARC_get_Mono_Grating_Installed Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Grating As Long) As Integer

'Monochromator Diverter Mirrors
Private Declare Function ARC_get_Mono_Diverter_Valid Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Diverter_Num As Long) As Integer
Private Declare Function ARC_get_Mono_Diverter_Pos Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Diverter_Num As Long, Diverter_Pos As Long) As Integer
Private Declare Function ARC_set_Mono_Diverter_Pos Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Diverter_Num As Long, ByVal Diverter_Pos As Long) As Integer
Private Declare Function ARC_get_Mono_Diverter_Pos_Var Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Diverter_Num As Long, Diverter_Pos As Variant) As Integer

'Monochromator Slits
Private Declare Function ARC_get_Mono_Slit_Type Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Slit_Pos As Long, Slit_Type As Long) As Integer
Private Declare Function ARC_get_Mono_Slit_Type_Var Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Slit_Pos As Long, Slit_Type As Variant) As Integer
Private Declare Function ARC_get_Mono_Slit_Width Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Slit_Pos As Long, Slit_Width As Long) As Integer
Private Declare Function ARC_set_Mono_Slit_Width Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Slit_Pos As Long, ByVal Slit_Width As Long) As Integer
Private Declare Function ARC_Mono_Slit_Home Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Slit_Pos As Long) As Integer
Private Declare Function ARC_Mono_Slit_Name_Var Lib "ARC_SpectraPro.dll" (ByVal Slit_Num As Long, Slit_Name_Var As Variant) As Integer

'Monochromator Filter Wheel (not available on all models)
Private Declare Function ARC_get_Mono_Filter_Present Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long) As Integer
Private Declare Function ARC_get_Mono_Filter_Position Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Position As Long) As Integer
Private Declare Function ARC_set_Mono_Filter_Position Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Position As Long) As Integer
Private Declare Function ARC_get_Mono_Filter_Min_Pos Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Position As Long) As Integer
Private Declare Function ARC_get_Mono_Filter_Max_Pos Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Position As Long) As Integer
Private Declare Function ARC_Mono_Filter_Home Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long) As Integer

'advanced functions
'Gear
Private Declare Function ARC_get_Mono_Int_Led_On Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long) As Integer
Private Declare Function ARC_set_Mono_Int_Led Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Led_State As Integer) As Integer
Private Declare Function ARC_get_Mono_Motor_Int Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long) As Integer
Private Declare Function ARC_get_Mono_Wheel_Int Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long) As Integer
Private Declare Function ARC_Mono_Move_Steps Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Num_Steps As Long) As Integer

'Grating Values
Private Declare Function ARC_get_Mono_Init_Grating Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Init_Grating As Long) As Integer
Private Declare Function ARC_set_Mono_Init_Grating Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Init_Grating As Long) As Integer
Private Declare Function ARC_get_Mono_Init_Wave_nm Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Init_Wave As Double) As Integer
Private Declare Function ARC_set_Mono_Init_Wave_nm Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Init_Wave As Double) As Integer
Private Declare Function ARC_get_Mono_Init_ScanRate_nm Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Init_ScanRate As Double) As Integer
Private Declare Function ARC_set_Mono_Init_ScanRate_nm Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Init_ScanRate As Double) As Integer
Private Declare Function ARC_get_Mono_Grating_Offset Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Grating As Long, Offset As Long) As Integer
Private Declare Function ARC_get_Mono_Grating_Gadjust Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Grating As Long, Gadjust As Long) As Integer
Private Declare Function ARC_set_Mono_Grating_Offset Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Grating As Long, ByVal Offset As Long) As Integer
Private Declare Function ARC_set_Mono_Grating_Gadjust Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Grating As Long, ByVal Gadjust As Long) As Integer
Private Declare Function ARC_Mono_Grating_Calc_Offset Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Grating As Long, ByVal Wave As Double, ByVal RefWave As Double, newOffset As Long) As Integer
Private Declare Function ARC_Mono_Grating_Calc_Gadjust Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Grating As Long, ByVal Wave As Double, ByVal RefWave As Double, newGadjust As Long) As Integer
Private Declare Function ARC_Mono_Grating_UnInstall Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Grating As Long) As Integer
Private Declare Function ARC_Mono_Grating_Install Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Grating As Long, ByVal Density As Long, ByVal Blaze As Variant, ByVal NMBlaze As Long, ByVal HOLBlaze As Long, ByVal MIRROR As Long) As Integer
Private Declare Function ARC_Mono_Reset Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long) As Integer
Private Declare Function ARC_Mono_Restore_Factory_Settings Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long) As Integer
Private Declare Function ARC_Mono_Save_Factory_Settings Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal password As Double) As Integer

'Scan Values
Private Declare Function ARC_get_Mono_Scan_Rate_nm_min Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Scan_Rate As Double) As Integer
Private Declare Function ARC_set_Mono_Scan_Rate_nm_min Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Scan_Rate As Double) As Integer
Private Declare Function ARC_Mono_Start_Scan_To_nm Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Wavelength_nm As Double) As Integer
Private Declare Function ARC_Mono_Scan_Done Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Done_Moving As Long, Current_Wavelength_nm As Double) As Integer
' ARC_Mono_Start_Jog is use is limited in the VB development enviroment due to language/compiler limitations
Private Declare Function ARC_Mono_Start_Jog Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Jog_MaxRate As Long, ByVal JogUp As Long) As Integer
Private Declare Function ARC_Mono_Stop_Jog Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long) As Integer

' the current monochromator
Dim Mono_Enum As Long

Private Sub CalcGAdjBtn_Click()
Dim newWave    As Double
Dim newRefWave As Double
Dim newGAdj    As Long
Dim newStr     As String
Dim GratNum    As Long

' this is done using VB's error handling, catches severe typo's in the GratEdt field
On Error GoTo ConversionErrorHandler
' get the grating number
GratNum = GratEdt.Text
newWave = DisWaveEdt.Text
newRefWave = RefWaveEdt.Text
List1.Clear
If ARC_Mono_Grating_Calc_Gadjust(Mono_Enum, GratNum, newWave, newRefWave, newGAdj) = 0 Then
   ' Failed to set the scan rate
   List1.AddItem ("Error : Failed to Calculate GAdjust")
   Else 'Display the new GAdjust
   newStr = newGAdj
   GAdjLabel.Caption = "GAdj : " + newStr
   End If
Exit Sub

ConversionErrorHandler:  ' an error occured converting data to a usable value
End Sub

Private Sub CalcOffsetBtn_Click()
Dim newWave    As Double
Dim newRefWave As Double
Dim newOffset  As Long
Dim newStr     As String
Dim GratNum    As Long

' this is done using VB's error handling, catches severe typo's in the GratEdt field
On Error GoTo ConversionErrorHandler
' get the grating number
GratNum = GratEdt.Text
newWave = DisWaveEdt.Text
newRefWave = RefWaveEdt.Text
List1.Clear
If ARC_Mono_Grating_Calc_Offset(Mono_Enum, GratNum, newWave, newRefWave, newOffset) = 0 Then
   ' Failed to set the scan rate
   List1.AddItem ("Error : Failed to Calculate Offset")
   Else 'Display the new Offset
   newStr = newOffset
   OffsetLabel.Caption = "Offset : " + newStr
   End If
Exit Sub

ConversionErrorHandler:  ' an error occured converting data to a usable value
End Sub

Private Sub CloseBtn_Click()
If ARC_Close_Mono(Mono_Enum) <> 0 Then
   List1.Clear
   List1.AddItem ("Mono closed")
   Else
   List1.Clear
   List1.AddItem ("Error : Mono Close")
   End If
End Sub


Private Sub EnumEdt_Change()
' this is done using VB's error handling, catches severe typo's in the EnumEdt feild
On Error GoTo ConversionErrorHandler
' Store the latest Mono_Enum value ... this allows support of multiple Monochromators
Mono_Enum = EnumEdt.Text
' indicate if the current Enum is open and therefore valid
If ARC_Valid_Mono_Enum(Mono_Enum) <> 0 Then
   OpenChk.Value = 1 ' The device is open
   Else
   OpenChk.Value = 0 ' The Device is not open, or invalid
   End If
Exit Sub

ConversionErrorHandler:  ' an error occured converting data to a usable value

End Sub

Private Sub FindMonoBtn_Click()
' Search for Monochromators in the system
Dim Mono_Count As Integer
Dim TempStr As String

' Search for Monochromators
If ARC_Search_For_Mono(Num_Found) <> 0 Then
   ' List the Index and Model of each Monochromator found
   List1.Clear
   Mono_Count = 0
   
   Do While Mono_Count < Num_Found
      TempStr = Mono_Count
      TempStr = TempStr + " Index : "
      ' Get the Monochromator Model
      If ARC_get_Mono_preOpen_Model(Mono_Count, Model_Var) <> 0 Then
         TempStr = TempStr + Model_Var
         End If
      ' Add the Monochromator to the list
      List1.AddItem (TempStr)
      ' increment the index number
      Mono_Count = Mono_Count + 1
      Loop
   Else
   ' No Monochromators where found
   List1.Clear
   List1.AddItem ("Error : No Monochromators Found")
   End If
End Sub

Private Sub Form_Load()
' On load verify the ARC_SpectraPro.dll loaded by displaying it's Version
Dim TempStr1 As String
Dim TempStr2 As String

' get the dll version
TempStr1 = "ARC_SpectraPro.dll ver. "
ThrowAway = ARC_Ver(Majorval, Minorval, Buildval)
TempStr2 = Majorval
TempStr1 = TempStr1 + TempStr2 + "."
TempStr2 = Minorval
TempStr1 = TempStr1 + TempStr2 + "."
TempStr2 = Buildval
TempStr1 = TempStr1 + TempStr2
' Display the DLL version
DLLLabel.Caption = TempStr1
Form1.Caption = TempStr1

End Sub

Private Sub InstallGratBtn_Click()
Dim Density  As Long
Dim Blaze    As Variant
Dim NMBlaze  As Long
Dim HOLBlaze As Long
Dim MIRROR   As Long
Dim GratNum As Integer

' this is done using VB's error handling, catches severe typo's in the GratEdt field
On Error GoTo ConversionErrorHandler
' get the grating number
GratNum = GratEdt.Text
' get the new grating information
Density = GrooveEdt.Text
Blaze = BlazeEdt.Text
' default values
   MIRROR = 0
   NMBlaze = 0
   HOLBlaze = 0
If MirrorRadio.Value Then
   MIRROR = 1
   NMBlaze = 0
   HOLBlaze = 0
   End If
If nmRadio.Value Then
   MIRROR = 0
   NMBlaze = 1
   HOLBlaze = 0
   End If
If umRadio.Value Then
   MIRROR = 0
   NMBlaze = 0
   HOLBlaze = 0
   End If
If HOLRadio.Value Then
   MIRROR = 0
   NMBlaze = 0
   HOLBlaze = 1
   End If
List1.Clear
If ARC_Mono_Grating_Install(Mono_Enum, GratNum, Density, Blaze, NMBlaze, HOLBlaze, MIRROR) = 0 Then
   ' Failed to set the scan rate
   List1.AddItem ("Error : Unable to install grating")
   End If
Exit Sub

ConversionErrorHandler:  ' an error occured converting data to a usable value
End Sub


Private Sub IntLedChk_Click()
List1.Clear
If IntLedChk.Value = 0 Then
   If ARC_set_Mono_Int_Led(Mono_Enum, 0) = 0 Then
      List1.AddItem ("Error : Failed to turn off Int LED")
      End If
Else
   If ARC_set_Mono_Int_Led(Mono_Enum, 1) = 0 Then
      List1.AddItem ("Error : Failed to turn on Int LED")
      End If
End If
End Sub


Private Sub MonoInfoBtn_Click()
Dim outvar As Variant
Dim outstr As String
Dim outstr2 As String
Dim outdbl As Double
Dim outint As Long
Dim outint2 As Long
Dim gen_loop As Long
Dim loop_str As String


List1.Clear
If ARC_get_Mono_Model(Mono_Enum, outvar) <> 0 Then
      List1.AddItem ("Model : " + outvar)
Else
      List1.AddItem ("Error : No Model")
End If
If ARC_get_Mono_Serial(Mono_Enum, outvar) <> 0 Then
      List1.AddItem ("Serial : " + outvar)
Else
      List1.AddItem ("Error : No Serial")
End If
'// grating info
outint2 = 0
gen_loop = 1
Do Until gen_loop > 15
   If (ARC_get_Mono_Grating_Offset(Mono_Enum, gen_loop, outint) <> 0) And (ARC_get_Mono_Grating_Gadjust(Mono_Enum, gen_loop, outint2) <> 0) Then
      loop_str = gen_loop
      outstr = outint
      outstr2 = outint2
      List1.AddItem ("Grating" + loop_str + " Offset : " + outstr + " Gadjust : " + outstr2)
      End If
   gen_loop = gen_loop + 1
   Loop
'// mono startup settings
If ARC_get_Mono_Init_Grating(Mono_Enum, outint) <> 0 Then
   outstr = outint
   List1.AddItem ("Init Grating : " + outstr)
   Else
   List1.AddItem ("Error : Init Grating")
   End If
If ARC_get_Mono_Init_Wave_nm(Mono_Enum, outdbl) <> 0 Then
   outstr = outdbl
   List1.AddItem ("Init Wavelength : " + outstr)
   Else
   List1.AddItem ("Error : Init Wavelength")
   End If
If ARC_get_Mono_Init_ScanRate_nm(Mono_Enum, outdbl) <> 0 Then
   outstr = outdbl
   List1.AddItem ("Init Scan Rate : " + outstr)
   Else
   List1.AddItem ("Error : Init ScanRate")
   End If
If ARC_get_Mono_Scan_Rate_nm_min(Mono_Enum, outdbl) <> 0 Then
   outstr = outdbl
   List1.AddItem ("Scan Rate : " + outstr)
   Else
   List1.AddItem ("Error : Scan Rate nm/min")
   End If
'// mono interrupter settings
If ARC_get_Mono_Int_Led_On(Mono_Enum) <> 0 Then
     List1.AddItem ("Int Led : On")
     If ARC_get_Mono_Motor_Int(Mono_Enum) <> 0 Then
        List1.AddItem ("Motor Int : Open")
        Else
        List1.AddItem ("Motor Int : Closed")
        End If
     If ARC_get_Mono_Wheel_Int(Mono_Enum) <> 0 Then
        List1.AddItem ("Wheel Int : Open")
        Else
        List1.AddItem ("Wheel Int : Closed")
        End If
   Else
   List1.AddItem ("Int Led : Off")
   End If
End Sub

Private Sub MoveStepsBtn_Click()
Dim NumSteps As Long
Dim NumStr As String
Dim Old_nm As Double
Dim New_nm As Double
Dim Old_Str As String
Dim New_Str As String
Dim Dummy As Integer
' this is done using VB's error handling, catches severe typo's in the InitGratEdt field
On Error GoTo ConversionErrorHandler
NumSteps = StepsEdt.Text
' obtain the current wavelength in nm before we move, so we can display it later
Dummy = ARC_get_Mono_Wavelength_nm(Mono_Enum, Old_nm)
If ARC_Mono_Move_Steps(Mono_Enum, NumSteps) = 0 Then
   ' Failed to move number of steps
   List1.Clear
   List1.AddItem ("Error : Invalid Init Grating")
   Else
   ' moved required steps
   List1.Clear
   Old_Str = Old_nm
   List1.AddItem ("PreMove Wavelength : " + Old_Str + " nm")
   NumStr = NumSteps
   List1.AddItem ("Steps Moved : " + NumStr)
   ' obtain the current wavelength
   Dummy = ARC_get_Mono_Wavelength_nm(Mono_Enum, New_nm)
   New_Str = New_nm
   List1.AddItem ("Post Move Wavelength : " + New_Str + " nm")
   End If
Exit Sub

ConversionErrorHandler:  ' an error occured converting data to a usable value

End Sub

Private Sub OpenBtn_Click()
If ARC_Open_Mono(Mono_Enum, Mono_Enum) <> 0 Then
   OpenChk.Value = 1 ' The device is open
   Else
   OpenChk.Value = 0 ' The Device is not open, or invalid
   End If
End Sub

Private Sub ResetFactoryBtn_Click()
' Restore the Factory Default Settings
If ARC_Mono_Restore_Factory_Settings(Mono_Enum) = 0 Then
   List1.Clear
   List1.AddItem ("Error : Failed to Restore Factory Settings")
   Else
   List1.Clear
   End If
End Sub

Private Sub ResetMonoBtn_Click()
' Reset the MonoChromator
If ARC_Mono_Reset(Mono_Enum) = 0 Then
   List1.Clear
   List1.AddItem ("Error : Failed to Reset the Monochromator")
   Else
   List1.Clear
   End If
End Sub

Private Sub ScanBtn_Click()
Dim Start_Wave_nm As Double
Dim Stop_Wave_nm  As Double
Dim Scan_Done     As Long
Dim Cur_Wave      As Double
Dim NewRate       As Double
Dim Cur_Str       As String

' this is done using VB's error handling, catches severe typo's in the edit fields
On Error GoTo ConversionErrorHandler
' obtain the scan start and stop wavelength's
Start_Wave_nm = StartScanEdt.Text
Stop_Wave_nm = StopScanEdt.Text

List1.Clear
If ARC_set_Mono_Wavelength_nm(Mono_Enum, Start_Wave_nm) <> 0 Then 'set initial wavelength
   If ARC_Mono_Start_Scan_To_nm(Mono_Enum, Stop_Wave_nm) <> 0 Then 'start scanning
      Scan_Done = 0 ' pre-initialize the values
      Cur_Wave = 0#
      Do While (Scan_Done = 0) And (ARC_Mono_Scan_Done(Mono_Enum, Scan_Done, Cur_Wave) <> 0)
         'see if we are done, this function also updates the motor
         'speed so that it matches scan rate. It needs to be called
         'regularly until the scan is done.
         Cur_Str = Cur_Wave
         ScannmLabel.Caption = "Scan : " + Cur_Str + "nm"
         Loop
         
      ' needed to make sure the wavelength is correctly set, this function must
      ' be called at the end of a scan
      ARC_Mono_Stop_Jog (Mono_Enum)
      ' when done scanning read the final wavelength
      If ARC_get_Mono_Wavelength_nm(Mono_Enum, Cur_Wave) <> 0 Then
         Cur_Str = Cur_Wave
         ScannmLabel.Caption = "Scan : " + Cur_Str + "nm"
         Else
         List1.AddItem ("Error : Unable to Scan")
         End If
      End If
Else
   List1.AddItem ("Error : Unable to set Start Wavelength")
End If

Exit Sub

ConversionErrorHandler:  ' an error occured converting data to a usable value

End Sub

Private Sub SetGAdjBtn_Click()
Dim newWave    As Double
Dim newRefWave As Double
Dim newGAdj    As Long
Dim newStr     As String
Dim GratNum    As Long

' this is done using VB's error handling, catches severe typo's in the GratEdt field
On Error GoTo ConversionErrorHandler
' get the grating number
GratNum = GratEdt.Text
newWave = DisWaveEdt.Text
newRefWave = RefWaveEdt.Text
List1.Clear
If ARC_Mono_Grating_Calc_Gadjust(Mono_Enum, GratNum, newWave, newRefWave, newGAdj) = 0 Then
   ' Failed to set the scan rate
   List1.AddItem ("Error : Failed to Calculate GAdjust")
   Else 'Set the new GAdjust
   If ARC_set_Mono_Grating_Gadjust(Mono_Enum, GratNum, newGAdj) = 0 Then
      ' Failed to set a new GAdjust
      List1.AddItem ("Error : Failed to set GAdjust")
      End If
   End If
Exit Sub

ConversionErrorHandler:  ' an error occured converting data to a usable value
End Sub

Private Sub SetInitGratBtn_Click()
Dim NewGrating As Long
' this is done using VB's error handling, catches severe typo's in the InitGratEdt field
On Error GoTo ConversionErrorHandler
NewGrating = InitGratEdt.Text
' Change the grating
If ARC_set_Mono_Init_Grating(Mono_Enum, NewGrating) = 0 Then
   List1.Clear
   List1.AddItem ("Error : Invalid Init Grating")
   Else
   List1.Clear
   End If
Exit Sub

ConversionErrorHandler:  ' an error occured converting data to a usable value

End Sub

Private Sub SetInitScanBtn_Click()
Dim NewScan As Double
' this is done using VB's error handling, catches severe typo's in the InitScanEdt field
On Error GoTo ConversionErrorHandler
NewScan = InitScanEdt.Text
' Change the init scan rate
If ARC_set_Mono_Init_ScanRate_nm(Mono_Enum, NewScan) = 0 Then
   List1.Clear
   List1.AddItem ("Error : Invalid Init Scan Rate")
   Else
   List1.Clear
   End If
Exit Sub

ConversionErrorHandler:  ' an error occured converting data to a usable value

End Sub

Private Sub SetInitWaveBtn_Click()
Dim newWave As Double
' this is done using VB's error handling, catches severe typo's in the InitWaveEdt field
On Error GoTo ConversionErrorHandler
newWave = InitWaveEdt.Text
' Change the init wavelength
If ARC_get_Mono_Init_Wave_nm(Mono_Enum, newWave) = 0 Then
   List1.Clear
   List1.AddItem ("Error : Invalid Init Wave")
   Else
   List1.Clear
   End If
Exit Sub

ConversionErrorHandler:  ' an error occured converting data to a usable value

End Sub

Private Sub SetOffsetBtn_Click()
Dim newWave    As Double
Dim newRefWave As Double
Dim newOffset  As Long
Dim newStr     As String
Dim GratNum    As Long

' this is done using VB's error handling, catches severe typo's in the GratEdt field
On Error GoTo ConversionErrorHandler
' get the grating number
GratNum = GratEdt.Text
newWave = DisWaveEdt.Text
newRefWave = RefWaveEdt.Text
List1.Clear
If ARC_Mono_Grating_Calc_Offset(Mono_Enum, GratNum, newWave, newRefWave, newOffset) = 0 Then
   ' Failed to Calculate Offset
   List1.AddItem ("Error : Failed to Calculate Offset")
   Else 'Set the new Offset
   If ARC_set_Mono_Grating_Offset(Mono_Enum, GratNum, newOffset) = 0 Then
      ' Failed to set a new offset
      List1.AddItem ("Error : Failed to set Offset")
      End If
   End If
Exit Sub

ConversionErrorHandler:  ' an error occured converting data to a usable value
End Sub

Private Sub SetRateBtn_Click()
Dim NewRate As Double
' this is done using VB's error handling, catches severe typo's in the NewRate field
On Error GoTo ConversionErrorHandler
NewRate = ScanRateEdt.Text
   List1.Clear
If ARC_set_Mono_Scan_Rate_nm_min(Mono_Enum, NewRate) = 0 Then
   ' Failed to set the scan rate
   List1.AddItem ("Error : Invalid Set Scan Rate")
   End If
Exit Sub

ConversionErrorHandler:  ' an error occured converting data to a usable value

End Sub

Private Sub UnInstallBtn_Click()
Dim GratNum As Integer

' this is done using VB's error handling, catches severe typo's in the GratEdt field
On Error GoTo ConversionErrorHandler
' get the grating number
GratNum = GratEdt.Text
List1.Clear
If ARC_Mono_Grating_UnInstall(Mono_Enum, GratNum) = 0 Then
   ' Failed to set the scan rate
   List1.AddItem ("Error : Failed to Uninstall Grating")
   End If
Exit Sub

ConversionErrorHandler:  ' an error occured converting data to a usable value

End Sub
