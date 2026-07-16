VERSION 5.00
Begin VB.Form Form1 
   Caption         =   "Form1"
   ClientHeight    =   1905
   ClientLeft      =   60
   ClientTop       =   345
   ClientWidth     =   3540
   LinkTopic       =   "Form1"
   ScaleHeight     =   1905
   ScaleWidth      =   3540
   StartUpPosition =   3  'Windows Default
   Begin VB.CommandButton SetWaveBtn 
      Caption         =   "Set Wavelength"
      Height          =   495
      Left            =   1800
      TabIndex        =   4
      Top             =   1080
      Width           =   1335
   End
   Begin VB.TextBox WaveEdt 
      Height          =   375
      Left            =   120
      TabIndex        =   3
      Text            =   "WaveEdt"
      Top             =   1080
      Width           =   975
   End
   Begin VB.Label Label1 
      Caption         =   "nm"
      Height          =   375
      Left            =   1200
      TabIndex        =   5
      Top             =   1080
      Width           =   975
   End
   Begin VB.Label SerialLabel 
      Caption         =   "Serial :"
      Height          =   375
      Left            =   240
      TabIndex        =   2
      Top             =   480
      Width           =   2415
   End
   Begin VB.Label ModelLabel 
      Caption         =   "Model : "
      Height          =   375
      Left            =   240
      TabIndex        =   1
      Top             =   240
      Width           =   2295
   End
   Begin VB.Label DLLLabel 
      Caption         =   "ARC_SpectraPro.dll ver. "
      Height          =   375
      Left            =   0
      TabIndex        =   0
      Top             =   0
      Width           =   2775
   End
End
Attribute VB_Name = "Form1"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = False
' ARC_SpectraPro.DLL functions
Private Declare Function ARC_Ver Lib "ARC_SpectraPro.dll" (Majorval As Long, Minorval As Long, Buildval As Long) As Integer
Private Declare Function ARC_Search_For_Mono Lib "ARC_SpectraPro.dll" (Num_Found As Long) As Integer
Private Declare Function ARC_Open_Mono Lib "ARC_SpectraPro.dll" (ByVal Enum_Num As Long, Mono_Enum As Long) As Integer
Private Declare Function ARC_Close_Mono Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long) As Integer
Private Declare Function ARC_get_Mono_Model Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Model_str As Variant) As Integer
Private Declare Function ARC_get_Mono_Wavelength_nm Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Wavelength As Double) As Integer
Private Declare Function ARC_set_Mono_Wavelength_nm Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, ByVal Wavelength As Double) As Integer
Private Declare Function ARC_get_Mono_Serial Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Serial_str As Variant) As Integer

' Mono Enum value
Dim Mono_Enum As Long


Private Sub Form_Load()
Dim TempStr1 As String
Dim TempStr2 As String

' Initially set the Mono_Enum to a negative value
Mono_Enum = -1

' get the dll version
TempStr1 = "ARC_SpectraPro.dll ver. "
Throwaway = ARC_Ver(Majorval, Minorval, Buildval)
TempStr2 = Majorval
TempStr1 = TempStr1 + TempStr2 + "."
TempStr2 = Minorval
TempStr1 = TempStr1 + TempStr2 + "."
TempStr2 = Buildval
TempStr1 = TempStr1 + TempStr2
' Display the DLL version
DLLLabel.Caption = TempStr1

'Find if there is an attached monochromator
If ARC_Search_For_Mono(Num_Found) <> 0 Then
   'Open Attached Monochromator
   If ARC_Open_Mono(0, Mono_Enum) <> 0 Then
      'Display Monochromator Model on title
      Throwaway = ARC_get_Mono_Model(Mono_Enum, Model_str)
      TempStr1 = Model_str
      ModelLabel.Caption = "Model : " + TempStr1
      Form1.Caption = "Model : " + TempStr1
      ' Display Monochromator Serial Number
      Throwaway = ARC_get_Mono_Serial(Mono_Enum, Serial_str)
      TempStr1 = Serial_str
      SerialLabel.Caption = "Serial : " + TempStr1
      'Display Monochromator Wavelength
      If ARC_get_Mono_Wavelength_nm(Mono_Enum, Wavelength) <> 0 Then
         WaveEdt.Text = Wavelength
         Else
         WaveEdt.Text = "Failed to Read"
         End If
      End If
End If
      
End Sub

Private Sub Form_Unload(Cancel As Integer)
' close the port the monochromator is attached to
Throwaway = ARC_Close_Mono(Mono_Enum)
End Sub

Private Sub SetWaveBtn_Click()
Dim newWave As Double

' this is done using VB's error handling, catches severe typo's in the WaveEdt feild
On Error GoTo ConversionErrorHandler
newWave = WaveEdt.Text
' set the wavelength
Throwaway = ARC_set_Mono_Wavelength_nm(Mono_Enum, newWave)
' we ignore the result because we read back the wavelength in the next step
' read back the wavelength
If ARC_get_Mono_Wavelength_nm(Mono_Enum, Wavelength) <> 0 Then
   WaveEdt.Text = Wavelength
   Else
   WaveEdt.Text = "Failed to Read"
   End If
' we are done
Exit Sub

ConversionErrorHandler:  ' an error occured converting data to a usable value


End Sub
