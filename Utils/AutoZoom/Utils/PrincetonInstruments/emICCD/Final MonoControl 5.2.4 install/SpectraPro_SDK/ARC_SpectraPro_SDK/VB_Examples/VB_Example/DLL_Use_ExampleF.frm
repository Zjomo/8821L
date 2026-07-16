VERSION 5.00
Begin VB.Form Form1 
   Caption         =   "Form1"
   ClientHeight    =   5775
   ClientLeft      =   60
   ClientTop       =   345
   ClientWidth     =   13125
   LinkTopic       =   "Form1"
   ScaleHeight     =   5775
   ScaleWidth      =   13125
   StartUpPosition =   3  'Windows Default
   Begin VB.TextBox CenterEdt 
      Height          =   375
      Left            =   11520
      TabIndex        =   14
      Top             =   1200
      Width           =   855
   End
   Begin VB.CommandButton SetWaveBtn 
      Caption         =   "Set Wavelength"
      Height          =   375
      Left            =   9600
      TabIndex        =   27
      Top             =   1680
      Width           =   1815
   End
   Begin VB.OptionButton relCMradio 
      Caption         =   "Relative WaveNumber"
      Height          =   255
      Left            =   9480
      TabIndex        =   33
      Top             =   1320
      Width           =   2055
   End
   Begin VB.OptionButton abscmradio 
      Caption         =   "Absolute WaveNumber"
      Height          =   255
      Left            =   9480
      TabIndex        =   32
      Top             =   1080
      Width           =   2055
   End
   Begin VB.OptionButton evradio 
      Caption         =   "eV"
      Height          =   255
      Left            =   9480
      TabIndex        =   31
      Top             =   840
      Width           =   1695
   End
   Begin VB.OptionButton micronradio 
      Caption         =   "micron"
      Height          =   255
      Left            =   9480
      TabIndex        =   30
      Top             =   600
      Width           =   1695
   End
   Begin VB.OptionButton nmradio 
      Caption         =   "nm"
      Height          =   255
      Left            =   9480
      TabIndex        =   29
      Top             =   360
      Width           =   1695
   End
   Begin VB.OptionButton angradio 
      Caption         =   "ang"
      Height          =   255
      Left            =   9480
      TabIndex        =   28
      Top             =   120
      Width           =   1695
   End
   Begin VB.ListBox List1 
      Height          =   5715
      Left            =   3840
      TabIndex        =   21
      Top             =   0
      Width           =   4335
   End
   Begin VB.TextBox RawCMDEdt 
      Height          =   375
      Left            =   0
      TabIndex        =   20
      Top             =   3600
      Width           =   3855
   End
   Begin VB.TextBox FilterEdt 
      Height          =   375
      Left            =   9360
      TabIndex        =   19
      Top             =   3960
      Width           =   855
   End
   Begin VB.TextBox SlitWidthEdt 
      Height          =   375
      Left            =   9360
      TabIndex        =   18
      Top             =   3480
      Width           =   855
   End
   Begin VB.TextBox SlitNumEdt 
      Height          =   375
      Left            =   8400
      TabIndex        =   17
      Top             =   3480
      Width           =   855
   End
   Begin VB.TextBox GratEdt 
      Height          =   375
      Left            =   9360
      TabIndex        =   16
      Top             =   2640
      Width           =   855
   End
   Begin VB.TextBox TurEdt 
      Height          =   375
      Left            =   9360
      TabIndex        =   15
      Top             =   2160
      Width           =   855
   End
   Begin VB.TextBox WaveEdt 
      Height          =   375
      Left            =   8280
      TabIndex        =   13
      Top             =   840
      Width           =   855
   End
   Begin VB.CommandButton HomeFilterBtn 
      Caption         =   "Home Filter"
      Height          =   375
      Left            =   11760
      TabIndex        =   12
      Top             =   3960
      Width           =   1215
   End
   Begin VB.CommandButton FilterPosBtn 
      Caption         =   "Set Filter"
      Height          =   375
      Left            =   10320
      TabIndex        =   11
      Top             =   3960
      Width           =   1215
   End
   Begin VB.CommandButton HomeSlitBtn 
      Caption         =   "Home Slit"
      Height          =   375
      Left            =   11760
      TabIndex        =   10
      Top             =   3480
      Width           =   1215
   End
   Begin VB.CommandButton SlitWidthBtn 
      Caption         =   "Set Slit Width"
      Height          =   375
      Left            =   10320
      TabIndex        =   9
      Top             =   3480
      Width           =   1215
   End
   Begin VB.CommandButton SetGratEdt 
      Caption         =   "Set Grating"
      Height          =   375
      Left            =   10320
      TabIndex        =   8
      Top             =   2640
      Width           =   1215
   End
   Begin VB.CommandButton SetTurBtn 
      Caption         =   "Set Turret"
      Height          =   375
      Left            =   10320
      TabIndex        =   7
      Top             =   2160
      Width           =   1215
   End
   Begin VB.CommandButton RawCMDBtn 
      Caption         =   "Raw CMD"
      Height          =   375
      Left            =   0
      TabIndex        =   6
      Top             =   4080
      Width           =   1215
   End
   Begin VB.CommandButton MonoInfoBtn 
      Caption         =   "Display Mono State"
      Height          =   375
      Left            =   720
      TabIndex        =   5
      Top             =   1800
      Width           =   1815
   End
   Begin VB.CommandButton CloseBtn 
      Caption         =   "Close Enum"
      Height          =   375
      Left            =   960
      TabIndex        =   4
      Top             =   1320
      Width           =   1215
   End
   Begin VB.CommandButton OpenBtn 
      Caption         =   "Open Enum"
      Height          =   375
      Left            =   960
      TabIndex        =   3
      Top             =   840
      Width           =   1215
   End
   Begin VB.TextBox EnumEdt 
      Height          =   375
      Left            =   0
      TabIndex        =   2
      Text            =   "Enum "
      Top             =   840
      Width           =   855
   End
   Begin VB.CommandButton FindMonoBtn 
      Caption         =   "Find Mono(s)"
      Height          =   375
      Left            =   0
      TabIndex        =   1
      Top             =   360
      Width           =   1215
   End
   Begin VB.CheckBox OpenChk 
      Caption         =   "Mono Open"
      Height          =   375
      Left            =   2280
      TabIndex        =   22
      Top             =   840
      Width           =   1575
   End
   Begin VB.Label RawCmdLabel 
      Caption         =   "Result :"
      Height          =   1575
      Left            =   1320
      TabIndex        =   26
      Top             =   4080
      Width           =   2535
   End
   Begin VB.Label CMDLabel 
      Caption         =   "Send Raw Mono Command"
      Height          =   375
      Left            =   0
      TabIndex        =   25
      Top             =   3240
      Width           =   3615
   End
   Begin VB.Label Label2 
      Caption         =   "Center nm"
      Height          =   255
      Left            =   12360
      TabIndex        =   24
      Top             =   1320
      Width           =   735
   End
   Begin VB.Label WaveLabel 
      Caption         =   "New Wavelength"
      Height          =   375
      Left            =   8280
      TabIndex        =   23
      Top             =   360
      Width           =   975
   End
   Begin VB.Label DLLLabel 
      Caption         =   "ARC_SpectraPro.dll ver. "
      Height          =   255
      Left            =   0
      TabIndex        =   0
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

' the current monochromator
Dim Mono_Enum As Long


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

Private Sub FilterPosBtn_Click()
'This function only works with motorized filterwheels, and is not available on all models
Dim NewFilter As Long

' this is done using VB's error handling, catches severe typo's in the Edit field
On Error GoTo ConversionErrorHandler
NewFilter = FilterEdt.Text
' Change the Filter position
If ARC_set_Mono_Filter_Position(Mono_Enum, NewFilter) = 0 Then
   List1.Clear
   List1.AddItem ("Error : Invalid Filter Operation")
   Else
   List1.Clear
   End If

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

Private Sub HomeFilterBtn_Click()
'This function only works with motorized filterwheels, and is not available on all models

' Home the Filter position
If ARC_Mono_Filter_Home(Mono_Enum) = 0 Then
   List1.Clear
   List1.AddItem ("Error : Invalid Filter Home Operation")
   Else
   List1.Clear
   End If
End Sub

Private Sub HomeSlitBtn_Click()
'This function only works with motorized slits, all other slits will return an error
Dim NewSlit As Long

' this is done using VB's error handling, catches severe typo's in the Edit field
On Error GoTo ConversionErrorHandler
NewSlit = SlitNumEdt.Text
' Home the motorized slit
If ARC_Mono_Slit_Home(Mono_Enum, NewSlit) = 0 Then
   List1.Clear
   List1.AddItem ("Error : Invalid Slit Home Operation")
   Else
   List1.Clear
   End If
Exit Sub

ConversionErrorHandler:  ' an error occured converting data to a usable value
End Sub

Private Sub Label1_Click()

End Sub

Private Sub MonoInfoBtn_Click()
Dim outvar As Variant
Dim outstr As String
Dim outstr2 As String
Dim outdbl As Double
Dim outint As Long
Dim outbool As Integer
Dim LoopVal As Long
Dim wkstr As String

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
If ARC_get_Mono_Focallength(Mono_Enum, outdbl) <> 0 Then
   outstr = outdbl
   List1.AddItem ("FocalLength : " + outstr)
   Else
   List1.AddItem ("Error : No Focal Length")
   End If
If ARC_get_Mono_HalfAngle(Mono_Enum, outdbl) <> 0 Then
   outstr = outdbl
   List1.AddItem ("Half Angle : " + outstr)
   Else
   List1.AddItem ("Error : No HalfAngle")
   End If
If ARC_get_Mono_DetectorAngle(Mono_Enum, outdbl) <> 0 Then
   outstr = outdbl
   List1.AddItem ("Det Angle : " + outstr)
   Else
   List1.AddItem ("Error : No Detector Angle")
   End If
If ARC_get_Mono_Turret_Gratings(Mono_Enum, outint) <> 0 Then
   outstr = outint
   List1.AddItem ("Gratings per Turret : " + outstr)
   Else
   List1.AddItem ("Error : Gratings per Turret")
   End If
If ARC_get_Mono_Double(Mono_Enum, outbool) <> 0 Then
   If outbool <> 0 Then
        List1.AddItem ("Double Mono : True")
        Else
        List1.AddItem ("Double Mono : False")
        End If
   Else
   List1.AddItem ("Error : Double")
   End If
If ARC_get_Mono_Wavelength_nm(Mono_Enum, outdbl) <> 0 Then
   outstr = outdbl
   List1.AddItem ("Wavelength : " + outstr + " nm")
   Else
   List1.AddItem ("Error : Wavelength nm")
   End If
If ARC_get_Mono_Wavelength_ang(Mono_Enum, outdbl) <> 0 Then
   outstr = outdbl
   List1.AddItem ("Wavelength : " + outstr + " Angstrom")
   Else
   List1.AddItem ("Error : Wavelength Angstrom")
   End If
If ARC_get_Mono_Wavelength_eV(Mono_Enum, outdbl) <> 0 Then
   outstr = outdbl
   List1.AddItem ("Wavelength : " + outstr + " eV")
   Else
   List1.AddItem ("Error : Wavelength eV")
   End If
If ARC_get_Mono_Wavelength_micron(Mono_Enum, outdbl) <> 0 Then
   outstr = outdbl
   List1.AddItem ("Wavelength : " + outstr + " microns")
   Else
   List1.AddItem ("Error : Wavelength micron")
   End If
If ARC_get_Mono_Wavelength_absCM(Mono_Enum, outdbl) <> 0 Then
   outstr = outdbl
   List1.AddItem ("Wavelength : " + outstr + " absCM-1")
   Else
   List1.AddItem ("Error : Wavelength Absolute WaveNumber")
   End If
If ARC_get_Mono_Wavelength_relCM(Mono_Enum, 546, outdbl) <> 0 Then
   outstr = outdbl
   List1.AddItem ("Wavelength : " + outstr + " relCM-1, Centered at 546.0 nm")
   Else
   List1.AddItem ("Error : Wavelength Relative WaveNumber")
   End If
If ARC_get_Mono_Turret(Mono_Enum, outint) <> 0 Then
   outstr = outint
   List1.AddItem ("Turret : " + outstr)
   Else
   List1.AddItem ("Error : Turret")
   End If
If ARC_get_Mono_Grating(Mono_Enum, outint) <> 0 Then
   outstr = outint
   List1.AddItem ("Grating : " + outstr)
   Else
   List1.AddItem ("Error : Grating")
   End If
If ARC_get_Mono_Wavelength_Cutoff_nm(Mono_Enum, outdbl) <> 0 Then
   outstr = outdbl
   List1.AddItem ("Max Wave : " + outstr + " nm")
   Else
   List1.AddItem ("Error : Max Wave")
   End If
If ARC_get_Mono_Wavelength_Min_nm(Mono_Enum, outdbl) <> 0 Then
   outstr = outdbl
   List1.AddItem ("Min Wave : " + outstr)
   Else
   List1.AddItem ("Error : Min Wave")
   End If
' Display Grating List
LoopVal = 1
Do Until LoopVal > 15
     If ARC_get_Mono_Grating_Installed(Mono_Enum, LoopVal) <> 0 Then
        outstr = LoopVal
        wkstr = "Grating " + outstr + " : "
        If (ARC_get_Mono_Grating_Density(Mono_Enum, LoopVal, outint) <> 0) And (ARC_get_Mono_Grating_Blaze(Mono_Enum, LoopVal, outstr) <> 0) Then
           outstr2 = outint
           List1.AddItem (wkstr + outstr2 + " g/mm, Blaze = " + outstr)
           Else
           List1.AddItem ("Error : " + wkstr)
           End If
        End If
    LoopVal = LoopVal + 1
    Loop
' Get Diverter Mirrors and Slits
LoopVal = 1
Do Until LoopVal > 2
     If ARC_get_Mono_Diverter_Pos_Var(Mono_Enum, loopvar, outvar) <> 0 Then
        outstr = loopvar
        List1.AddItem ("Mirror " + outstr + " on : " + outvar)
        Else
        List1.AddItem ("Error : Mirror not Motorized")
        End If
     ThrowAway = ARC_Mono_Slit_Name_Var((LoopVal * 2) - 1, outvar)
     wkstr = outvar
     If ARC_get_Mono_Slit_Type_Var(Mono_Enum, (LoopVal * 2) - 1, outvar) <> 0 Then
        List1.AddItem (wkstr + " Slit : " + outvar)
        Else
        List1.AddItem ("Error : " + wkstr)
        End If
     If ARC_get_Mono_Slit_Width(Mono_Enum, (LoopVal * 2) - 1, outint) <> 0 Then
        outstr = outint
        List1.AddItem ("Slit Witdh : " + outstr)
        End If
     ThrowAway = ARC_Mono_Slit_Name_Var((LoopVal * 2), outvar)
     wkstr = outvar
     If ARC_get_Mono_Slit_Type_Var(Mono_Enum, (LoopVal * 2), outvar) <> 0 Then
        List1.AddItem (wkstr + " Slit : " + outvar)
        Else
        List1.AddItem ("Error : " + wkstr)
        End If
    If ARC_get_Mono_Slit_Width(Mono_Enum, (LoopVal * 2), outint) <> 0 Then
       outstr = outint
       List1.AddItem ("Slit Witdh : " + outstr)
       End If
    LoopVal = LoopVal + 1
    Loop
ThrowAway = ARC_get_Mono_Double(Mono_Enum, outbool)
If outbool <> 0 Then ' slave present
   LoopVal = 3
   Do Until LoopVal > 4
      If ARC_get_Mono_Diverter_Pos_Var(Mono_Enum, loopvar, outvar) <> 0 Then
         outstr = loopvar
         List1.AddItem ("Mirror " + outstr + " on : " + outvar)
         Else
         List1.AddItem ("Error : Mirror not Motorized")
         End If
      ThrowAway = ARC_Mono_Slit_Name_Var((LoopVal * 2) - 1, outvar)
      wkstr = outvar
      If ARC_get_Mono_Slit_Type_Var(Mono_Enum, (LoopVal * 2) - 1, outvar) <> 0 Then
         List1.AddItem (wkstr + " Slit : " + outvar)
         Else
         List1.AddItem ("Error : " + wkstr)
         End If
      If ARC_get_Mono_Slit_Width(Mono_Enum, (LoopVal * 2) - 1, outint) <> 0 Then
         outstr = outint
         List1.AddItem ("Slit Witdh : " + outstr)
         End If
      ThrowAway = ARC_Mono_Slit_Name_Var((LoopVal * 2), outvar)
      wkstr = outvar
      If ARC_get_Mono_Slit_Type_Var(Mono_Enum, (LoopVal * 2), outvar) <> 0 Then
         List1.AddItem (wkstr + " Slit : " + outvar)
         Else
         List1.AddItem ("Error : " + wkstr)
         End If
      If ARC_get_Mono_Slit_Width(Mono_Enum, (LoopVal * 2), outint) <> 0 Then
         outstr = outint
         List1.AddItem ("Slit Witdh : " + outstr)
         End If
      LoopVal = LoopVal + 1
      Loop
   End If
' get filter info
If ARC_get_Mono_Filter_Present(Mono_Enum) <> 0 Then
   If ARC_get_Mono_Filter_Position(Mono_Enum, outint) <> 0 Then
      outstr = outint
      List1.AddItem ("Filter Position : " + outstr)
      Else
      List1.AddItem ("Error : Filter Position")
      End If
   Else
   List1.AddItem ("No Filter")
   End If
End Sub

Private Sub Form_Unload(Cancel As Integer)
Dim LoopVal As Long
' close all the ports a monochromator may be attached to
LoopVal = 0
Do Until LoopVal > 15
   ThrowAway = ARC_Close_Mono(LoopVal)
   LoopVal = LoopVal + 1
   Loop
End Sub

Private Sub OpenBtn_Click()
If ARC_Open_Mono(Mono_Enum, Mono_Enum) <> 0 Then
   OpenChk.Value = 1 ' The device is open
   Else
   OpenChk.Value = 0 ' The Device is not open, or invalid
   End If
End Sub

Private Sub OpenChk_Click()

End Sub

Private Sub RawCMDBtn_Click()
Dim SendCMD As Variant
Dim RCVCmd As Variant
' send a command to the instrument, note the function adds the required CR (^M)
' to the function, so the user does not need to. Placing a CR in the command
' string can have unpredictable results.
SendCMD = RawCMDEdt.Text
If ARC_Send_CMD_To_Mono(Mono_Enum, SendCMD, RCVCmd, 5000) = 0 Then
   List1.Clear
   List1.AddItem ("Error : Invalid Command")
   Else
   List1.Clear
   End If
RawCmdLabel.Caption = "Returned : " + RCVCmd
End Sub

Private Sub SetGratEdt_Click()
Dim NewGrat As Long
' this is done using VB's error handling, catches severe typo's in the TurEdt feild
On Error GoTo ConversionErrorHandler
' Store the latest Mono_Enum value ... this allows support of multiple Monochromator
NewGrat = GratEdt.Text
' Change the Turret
If ARC_set_Mono_Grating(Mono_Enum, NewGrat) = 0 Then
   List1.Clear
   List1.AddItem ("Error : Invalid Grating")
   Else
   List1.Clear
   End If
Exit Sub

ConversionErrorHandler:  ' an error occured converting data to a usable value

End Sub

Private Sub SetTurBtn_Click()
Dim NewTurret As Long
' this is done using VB's error handling, catches severe typo's in the TurEdt field
On Error GoTo ConversionErrorHandler
NewTurret = TurEdt.Text
' Change the Turret
If ARC_set_Mono_Turret(Mono_Enum, NewTurret) = 0 Then
   List1.Clear
   List1.AddItem ("Error : Invalid Turret")
   Else
   List1.Clear
   End If
Exit Sub

ConversionErrorHandler:  ' an error occured converting data to a usable value

End Sub

Private Sub SetWaveBtn_Click()
Dim newWave As Double
Dim newCenter As Double

' this is done using VB's error handling, catches severe typo's in the edit field(s)
On Error GoTo ConversionErrorHandler
' obtain the new Wavelength
newWave = WaveEdt.Text
' Decide which units to use and move to desired wavelength
If angradio.Value = True Then
   If ARC_set_Mono_Wavelength_ang(Mono_Enum, newWave) = 0 Then
      List1.Clear
      List1.AddItem ("Error : Invalid Wavelength (ang)")
      Else
      List1.Clear
      End If
   End If
If nmradio.Value = True Then
   If ARC_set_Mono_Wavelength_nm(Mono_Enum, newWave) = 0 Then
      List1.Clear
      List1.AddItem ("Error : Invalid Wavelength (nm)")
      Else
      List1.Clear
      End If
   End If
If micronradio.Value = True Then
   If ARC_set_Mono_Wavelength_micron(Mono_Enum, newWave) = 0 Then
      List1.Clear
      List1.AddItem ("Error : Invalid Wavelength (micron)")
      Else
      List1.Clear
      End If
   End If
If evradio.Value = True Then
   If ARC_set_Mono_Wavelength_eV(Mono_Enum, newWave) = 0 Then
      List1.Clear
      List1.AddItem ("Error : Invalid Wavelength (eV)")
      Else
      List1.Clear
      End If
   End If
If abscmradio.Value = True Then
   If ARC_set_Mono_Wavelength_absCM(Mono_Enum, newWave) = 0 Then
      List1.Clear
      List1.AddItem ("Error : Invalid Wavelength (CM-1)")
      Else
      List1.Clear
      End If
   End If
If relCMradio.Value = True Then
   newCenter = CenterEdt.Text
   If ARC_set_Mono_Wavelength_relCM(Mono_Enum, newCenter, newWave) = 0 Then
      List1.Clear
      List1.AddItem ("Error : Invalid Wavelength (CM-1)")
      Else
      List1.Clear
      End If
   End If
Exit Sub

ConversionErrorHandler:  ' an error occured converting data to a usable value

End Sub

Private Sub SlitWidthBtn_Click()
'This function only works with motorized slits, all other slits will return an error
Dim NewSlit As Long
Dim NewWidth As Long

' this is done using VB's error handling, catches severe typo's in the Edit field
On Error GoTo ConversionErrorHandler
NewSlit = SlitNumEdt.Text
NewWidth = SlitWidthEdt.Text
' Change the Slit Width
If ARC_set_Mono_Slit_Width(Mono_Enum, NewSlit, NewWidth) = 0 Then
   List1.Clear
   List1.AddItem ("Error : Invalid Slit Width Operation")
   Else
   List1.Clear
   End If
Exit Sub

ConversionErrorHandler:  ' an error occured converting data to a usable value
End Sub
