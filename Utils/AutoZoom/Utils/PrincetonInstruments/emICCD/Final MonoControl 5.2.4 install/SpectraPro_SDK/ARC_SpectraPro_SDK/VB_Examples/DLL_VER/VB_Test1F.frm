VERSION 5.00
Begin VB.Form Form1 
   Caption         =   "Form1"
   ClientHeight    =   3195
   ClientLeft      =   60
   ClientTop       =   345
   ClientWidth     =   4185
   LinkTopic       =   "Form1"
   ScaleHeight     =   3195
   ScaleWidth      =   4185
   StartUpPosition =   3  'Windows Default
   Begin VB.CommandButton Command1 
      Caption         =   "Get DLL Ver."
      Height          =   735
      Left            =   120
      TabIndex        =   0
      Top             =   240
      Width           =   1815
   End
   Begin VB.Label Label4 
      Caption         =   "ARC_SpectraPro.dll Version"
      Height          =   375
      Left            =   240
      TabIndex        =   4
      Top             =   1200
      Width           =   3855
   End
   Begin VB.Label Label3 
      Caption         =   "Build Ver."
      Height          =   495
      Left            =   240
      TabIndex        =   3
      Top             =   2520
      Width           =   2655
   End
   Begin VB.Label Label2 
      Caption         =   "Minor Ver."
      Height          =   375
      Left            =   240
      TabIndex        =   2
      Top             =   2040
      Width           =   2535
   End
   Begin VB.Label Label1 
      Caption         =   "Major Ver."
      Height          =   375
      Left            =   240
      TabIndex        =   1
      Top             =   1560
      Width           =   2415
   End
End
Attribute VB_Name = "Form1"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = False

Private Declare Function ARC_Ver Lib "ARC_SpectraPro.dll" (majorval As Long, Minorval As Long, Buildval As Long) As Integer




Private Sub Command1_Click()
Dim TempStr As String

' Read DLL Version using ARC_Ver()
ThroughAway = ARC_Ver(majorval, Minorval, Buildval)

' display the version to the user
TempStr = majorval
Label1.Caption = "Major Ver." + TempStr
TempStr = Minorval
Label2.Caption = "Minor Ver." + TempStr
TempStr = Buildval
Label3.Caption = "Build Ver." + TempStr

End Sub



