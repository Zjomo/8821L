VERSION 5.00
Begin VB.Form Form1 
   Caption         =   "Form1"
   ClientHeight    =   2865
   ClientLeft      =   60
   ClientTop       =   345
   ClientWidth     =   3690
   LinkTopic       =   "Form1"
   ScaleHeight     =   2865
   ScaleWidth      =   3690
   StartUpPosition =   3  'Windows Default
   Begin VB.TextBox Text1 
      Height          =   375
      Left            =   1800
      TabIndex        =   5
      Top             =   240
      Width           =   735
   End
   Begin VB.CommandButton Command1 
      Caption         =   "Open Port"
      Height          =   495
      Left            =   960
      TabIndex        =   4
      Top             =   720
      Width           =   1695
   End
   Begin VB.Label Label4 
      Caption         =   "Mono Enum : "
      Height          =   375
      Left            =   1200
      TabIndex        =   3
      Top             =   2280
      Width           =   2295
   End
   Begin VB.Label Label3 
      Caption         =   "Wavelength : --- nm"
      Height          =   375
      Left            =   1200
      TabIndex        =   2
      Top             =   1800
      Width           =   2295
   End
   Begin VB.Label Label2 
      Caption         =   "Model :"
      Height          =   375
      Left            =   1200
      TabIndex        =   1
      Top             =   1320
      Width           =   2295
   End
   Begin VB.Label Label1 
      Caption         =   "Com Port"
      Height          =   255
      Left            =   960
      TabIndex        =   0
      Top             =   360
      Width           =   735
   End
End
Attribute VB_Name = "Form1"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = False
' ARC_SpectraPro.DLL functions
Private Declare Function ARC_Ver Lib "ARC_SpectraPro.dll" (Majorval As Long, Minorval As Long, Buildval As Long) As Integer
Private Declare Function ARC_Open_Mono_Port Lib "ARC_SpectraPro.dll" (ByVal Com_Num As Long, Mono_Enum As Long) As Integer
Private Declare Function ARC_Close_Mono Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long) As Integer
Private Declare Function ARC_get_Mono_Model Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Model_str As Variant) As Integer
Private Declare Function ARC_get_Mono_Wavelength_nm Lib "ARC_SpectraPro.dll" (ByVal Mono_Enum As Long, Wavelength As Double) As Integer

' Mono Enum value
Dim Mono_Enum As Long
Dim Com_Num As Long


Private Sub Command1_Click()
Dim Model As Variant
Dim Wave As Double
Dim WaveStr As String
Dim EnumStr As String

' this is done using VB's error handling, catches severe typo's in the Text1 field
On Error GoTo ConversionErrorHandler
' obtain com port COMX
Com_Num = Text1.Text

If ARC_Open_Mono_Port(Com_Num, Mono_Enum) <> 0 Then
' found and opened a monochromator
' Display the Model
If ARC_get_Mono_Model(Mono_Enum, Model) <> 0 Then
   Label2.Caption = "Model : " + Model
   Else
   Label2.Caption = "Error : Model"
   End If
' Display the Wavelength
If ARC_get_Mono_Wavelength_nm(Mono_Enum, Wave) <> 0 Then
   WaveStr = Wave
   Label3.Caption = "Wavelength : " + WaveStr + " nm"
   Else
   Label3.Caption = "Error : Wavelength"
   End If
' Display our current Enumeration Value
EnumStr = Mono_Enum
Label4.Caption = "Mono Enum : " + EnumStr
' We are done with Monochromator, close it
If ARC_Close_Mono(Mono_Enum) <> 0 Then
   Mono_Enum = -1
   End If
Else
' failed to find a Monochromator on the COMX port
Label2.Caption = "Model : "
Label3.Caption = "Wavelength : --- nm"
Label4.Caption = "Mono Enum : "
End If

' we are done
Exit Sub

ConversionErrorHandler:  ' an error occured converting data to a usable value
End Sub

Private Sub Form_Load()

' Initially set the Mono_Enum to a negative value
Mono_Enum = -1

End Sub
