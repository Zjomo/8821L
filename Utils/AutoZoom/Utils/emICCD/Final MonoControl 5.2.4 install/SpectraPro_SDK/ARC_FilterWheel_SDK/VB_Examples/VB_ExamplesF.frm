VERSION 5.00
Begin VB.Form Form1 
   Caption         =   "Form1"
   ClientHeight    =   5745
   ClientLeft      =   60
   ClientTop       =   345
   ClientWidth     =   8205
   LinkTopic       =   "Form1"
   ScaleHeight     =   5745
   ScaleWidth      =   8205
   StartUpPosition =   3  'Windows Default
   Begin VB.CommandButton FilterPosBtn 
      Caption         =   "Set Filter"
      Height          =   375
      Left            =   960
      TabIndex        =   14
      Top             =   2520
      Width           =   1215
   End
   Begin VB.CommandButton HomeFilterBtn 
      Caption         =   "Home Filter"
      Height          =   375
      Left            =   2400
      TabIndex        =   13
      Top             =   2520
      Width           =   1215
   End
   Begin VB.TextBox FilterEdt 
      Height          =   375
      Left            =   0
      TabIndex        =   12
      Top             =   2520
      Width           =   855
   End
   Begin VB.CheckBox OpenChk 
      Caption         =   "FilterWheel Open"
      Height          =   375
      Left            =   2280
      TabIndex        =   8
      Top             =   840
      Width           =   1575
   End
   Begin VB.CommandButton FindFilterBtn 
      Caption         =   "Find Filter(s)"
      Height          =   375
      Left            =   0
      TabIndex        =   7
      Top             =   360
      Width           =   1215
   End
   Begin VB.TextBox EnumEdt 
      Height          =   375
      Left            =   0
      TabIndex        =   6
      Text            =   "Enum "
      Top             =   840
      Width           =   855
   End
   Begin VB.CommandButton OpenBtn 
      Caption         =   "Open Enum"
      Height          =   375
      Left            =   960
      TabIndex        =   5
      Top             =   840
      Width           =   1215
   End
   Begin VB.CommandButton CloseBtn 
      Caption         =   "Close Enum"
      Height          =   375
      Left            =   960
      TabIndex        =   4
      Top             =   1320
      Width           =   1215
   End
   Begin VB.CommandButton FilterInfoBtn 
      Caption         =   "Display Filter State"
      Height          =   375
      Left            =   720
      TabIndex        =   3
      Top             =   1800
      Width           =   1815
   End
   Begin VB.CommandButton RawCMDBtn 
      Caption         =   "Raw CMD"
      Height          =   375
      Left            =   0
      TabIndex        =   2
      Top             =   4080
      Width           =   1215
   End
   Begin VB.TextBox RawCMDEdt 
      Height          =   375
      Left            =   0
      TabIndex        =   1
      Top             =   3600
      Width           =   3855
   End
   Begin VB.ListBox List1 
      Height          =   5715
      Left            =   3840
      TabIndex        =   0
      Top             =   0
      Width           =   4335
   End
   Begin VB.Label DLLLabel 
      Caption         =   "ARC_FilterWheel.dll ver. "
      Height          =   255
      Left            =   0
      TabIndex        =   11
      Top             =   0
      Width           =   2895
   End
   Begin VB.Label CMDLabel 
      Caption         =   "Send Raw Mono Command"
      Height          =   375
      Left            =   0
      TabIndex        =   10
      Top             =   3240
      Width           =   3615
   End
   Begin VB.Label RawCmdLabel 
      Caption         =   "Result :"
      Height          =   1575
      Left            =   1320
      TabIndex        =   9
      Top             =   4080
      Width           =   2535
   End
End
Attribute VB_Name = "Form1"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = False
'ARC_FilterWheel.dll functions
'High level Communications
Private Declare Function ARC_Search_For_Filter Lib "ARC_FilterWheel.dll" (Num_Found As Long) As Integer
Private Declare Function ARC_Open_Filter Lib "ARC_FilterWheel.dll" (ByVal Enum_Num As Long, Filter_Enum As Long) As Integer
Private Declare Function ARC_Open_Filter_Port Lib "ARC_FilterWheel.dll" (ByVal Com_Num As Long, Filter_Enum As Long) As Integer
Private Declare Function ARC_Close_Filter Lib "ARC_FilterWheel.dll" (ByVal Filter_Enum As Long) As Integer
Private Declare Function ARC_Valid_Filter_Enum Lib "ARC_FilterWheel.dll" (ByVal Filter_Enum As Long) As Integer
Private Declare Function ARC_get_Filter_preOpen_Model Lib "ARC_FilterWheel.dll" (ByVal Filter_Enum As Long, Model_Var As Variant) As Integer
Private Declare Function ARC_Ver Lib "ARC_FilterWheel.dll" (Majorval As Long, Minorval As Long, Buildval As Long) As Integer
'Direct Communications Function
Private Declare Function ARC_Send_CMD_To_Filter Lib "ARC_FilterWheel.dll" (ByVal Filter_Enum As Long, ByVal SendCMD As Variant, RCV_Var As Variant, ByVal ms_Timeout As Long) As Integer

'FilterWheel Information
Private Declare Function ARC_get_Filter_Model Lib "ARC_FilterWheel.dll" (ByVal Filter_Enum As Long, Model_str As Variant) As Integer
Private Declare Function ARC_get_Filter_Serial Lib "ARC_FilterWheel.dll" (ByVal Filter_Enum As Long, Serial_str As Variant) As Integer

'Filter Wheel
Private Declare Function ARC_get_Filter_Present Lib "ARC_FilterWheel.dll" (ByVal Filter_Enum As Long) As Integer
Private Declare Function ARC_get_Filter_Position Lib "ARC_FilterWheel.dll" (ByVal Filter_Enum As Long, Position As Long) As Integer
Private Declare Function ARC_set_Filter_Position Lib "ARC_FilterWheel.dll" (ByVal Filter_Enum As Long, ByVal Position As Long) As Integer
Private Declare Function ARC_get_Filter_Min_Pos Lib "ARC_FilterWheel.dll" (ByVal Filter_Enum As Long, Position As Long) As Integer
Private Declare Function ARC_get_Filter_Max_Pos Lib "ARC_FilterWheel.dll" (ByVal Filter_Enum As Long, Position As Long) As Integer
Private Declare Function ARC_Filter_Home Lib "ARC_FilterWheel.dll" (ByVal Filter_Enum As Long) As Integer

' the current FilterWheel
Dim Filter_Enum As Long

Private Sub CloseBtn_Click()
If ARC_Close_Filter(Filter_Enum) <> 0 Then
   List1.Clear
   List1.AddItem ("Filter closed")
   Else
   List1.Clear
   List1.AddItem ("Error : Filter Close")
   End If
End Sub

Private Sub EnumEdt_Change()
' this is done using VB's error handling, catches severe typo's in the EnumEdt feild
On Error GoTo ConversionErrorHandler
' Store the latest Filter_Enum value ... this allows support of multiple Filters
Filter_Enum = EnumEdt.Text
' indicate if the current Enum is open and therefore valid
If ARC_Valid_Filter_Enum(Filter_Enum) <> 0 Then
   OpenChk.Value = 1 ' The device is open
   Else
   OpenChk.Value = 0 ' The Device is not open, or invalid
   End If
Exit Sub

ConversionErrorHandler:  ' an error occured converting data to a usable value

End Sub

Private Sub FilterInfoBtn_Click()
Dim outvar As Variant
Dim outstr As String
Dim outstr2 As String
Dim outdbl As Double
Dim outint As Long
Dim outbool As Integer
Dim LoopVal As Long
Dim wkstr As String

List1.Clear
If ARC_get_Filter_Model(Filter_Enum, outvar) <> 0 Then
   List1.AddItem ("Model : " + outvar)
   Else
   List1.AddItem ("Error : No Model")
   End If
If ARC_get_Filter_Serial(Filter_Enum, outvar) <> 0 Then
   List1.AddItem ("Serial : " + outvar)
   Else
   List1.AddItem ("Error : No Serial")
   End If
' get filter info
If ARC_get_Filter_Present(Filter_Enum) <> 0 Then
   If ARC_get_Filter_Position(Filter_Enum, outint) <> 0 Then
      outstr = outint
      List1.AddItem ("Filter Position : " + outstr)
      Else
      List1.AddItem ("Error : Filter Position")
      End If
   If ARC_get_Filter_Min_Pos(Filter_Enum, outint) <> 0 Then
      outstr = outint
      List1.AddItem ("Min Filter Position : " + outstr)
      Else
      List1.AddItem ("Error : Min Filter Position")
      End If
   If ARC_get_Filter_Max_Pos(Filter_Enum, outint) <> 0 Then
      outstr = outint
      List1.AddItem ("Max Filter Position : " + outstr)
      Else
      List1.AddItem ("Error : Max Filter Position")
      End If
   Else
   List1.AddItem ("No Filter")
   End If
End Sub

Private Sub FilterPosBtn_Click()
'This function only works with motorized filterwheels, and is not available on all models
Dim NewFilter As Long

' this is done using VB's error handling, catches severe typo's in the Edit field
On Error GoTo ConversionErrorHandler
NewFilter = FilterEdt.Text
' Change the Filter position
If ARC_set_Filter_Position(Filter_Enum, NewFilter) = 0 Then
   List1.Clear
   List1.AddItem ("Error : Invalid Filter Operation")
   Else
   List1.Clear
   End If

ConversionErrorHandler:  ' an error occured converting data to a usable value
End Sub

Private Sub FindFilterBtn_Click()
' Search for FilterWheels in the system
Dim Filter_Count As Integer
Dim TempStr As String

' Search for FilterWheels
If ARC_Search_For_Filter(Num_Found) <> 0 Then
   ' List the Index and Model of each FilterWheel found
   List1.Clear
   Filter_Count = 0
   
   Do While Filter_Count < Num_Found
      TempStr = Filter_Count
      TempStr = TempStr + " Index : "
      ' Get the FilterWheel Model
      If ARC_get_Filter_preOpen_Model(Filter_Count, Model_Var) <> 0 Then
         TempStr = TempStr + Model_Var
         End If
      ' Add the FilterWheel to the list
      List1.AddItem (TempStr)
      ' increment the index number
      Filter_Count = Filter_Count + 1
      Loop
   Else
   ' No Filterwheels where found
   List1.Clear
   List1.AddItem ("Error : No FilterWheels Found")
   End If
End Sub

Private Sub Form_Load()
' On load verify the ARC_SpectraPro.dll loaded by displaying it's Version
Dim TempStr1 As String
Dim TempStr2 As String

' get the dll version
TempStr1 = "ARC_FilterWheel.dll ver. "
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
If ARC_Filter_Home(Filter_Enum) = 0 Then
   List1.Clear
   List1.AddItem ("Error : Invalid Filter Home Operation")
   Else
   List1.Clear
   End If
End Sub

Private Sub OpenBtn_Click()
If ARC_Open_Filter(Filter_Enum, Filter_Enum) <> 0 Then
   OpenChk.Value = 1 ' The device is open
   Else
   OpenChk.Value = 0 ' The Device is not open, or invalid
   End If
End Sub

Private Sub RawCMDBtn_Click()
Dim SendCMD As Variant
Dim RCVCmd As Variant
' send a command to the instrument, note the function adds the required CR (^M)
' to the function, so the user does not need to. Placing a CR in the command
' string can have unpredictable results.
SendCMD = RawCMDEdt.Text
If ARC_Send_CMD_To_Filter(Filter_Enum, SendCMD, RCVCmd, 5000) = 0 Then
   List1.Clear
   List1.AddItem ("Error : Invalid Command")
   Else
   List1.Clear
   End If
RawCmdLabel.Caption = "Returned : " + RCVCmd
End Sub
