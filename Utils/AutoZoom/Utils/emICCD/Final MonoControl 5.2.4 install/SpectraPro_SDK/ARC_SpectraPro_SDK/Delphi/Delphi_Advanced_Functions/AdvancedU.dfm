object Advanced_FunctionsF: TAdvanced_FunctionsF
  Left = 190
  Top = 107
  Width = 759
  Height = 440
  HorzScrollBar.Visible = False
  VertScrollBar.Visible = False
  Caption = 'Advanced DLL Functions'
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'MS Sans Serif'
  Font.Style = []
  OldCreateOrder = False
  OnCreate = FormCreate
  OnDestroy = FormDestroy
  PixelsPerInch = 96
  TextHeight = 13
  object StepLabel: TLabel
    Left = 472
    Top = 35
    Width = 69
    Height = 13
    Caption = 'Steps to Move'
  end
  object JogWaveNMLabel: TLabel
    Left = 672
    Top = 64
    Width = 3
    Height = 13
    Caption = '.'
  end
  object ScanRateLabel: TLabel
    Left = 488
    Top = 112
    Width = 76
    Height = 13
    Caption = 'New Scan Rate'
  end
  object StartScanLabel: TLabel
    Left = 478
    Top = 136
    Width = 93
    Height = 13
    Caption = 'Scan Start Rate nm'
  end
  object StopScanLAbel: TLabel
    Left = 472
    Top = 160
    Width = 99
    Height = 13
    Caption = 'Scan Stop Wave nm'
  end
  object ScanWaveNMLabel: TLabel
    Left = 656
    Top = 144
    Width = 3
    Height = 13
    Caption = '.'
  end
  object GratLabel: TLabel
    Left = 472
    Top = 200
    Width = 59
    Height = 13
    Caption = 'Grating Num'
  end
  object GroovesLabel: TLabel
    Left = 472
    Top = 232
    Width = 67
    Height = 13
    Caption = 'Grooves / mm'
  end
  object BlazeStr: TLabel
    Left = 480
    Top = 256
    Width = 56
    Height = 13
    Caption = 'Blaze String'
  end
  object NewOffsetLabel: TLabel
    Left = 608
    Top = 344
    Width = 3
    Height = 13
    Caption = '.'
  end
  object NewGAdjLabel: TLabel
    Left = 608
    Top = 384
    Width = 3
    Height = 13
    Caption = '.'
  end
  object DisplayWaveLabel: TLabel
    Left = 472
    Top = 320
    Width = 66
    Height = 13
    Caption = 'Display Wave'
  end
  object RefWaveLabel: TLabel
    Left = 472
    Top = 360
    Width = 49
    Height = 13
    Caption = 'Ref Wave'
  end
  object ConnectedChk: TCheckBox
    Left = 144
    Top = 56
    Width = 97
    Height = 17
    Caption = 'Connected'
    TabOrder = 0
  end
  object FindPortBtn: TButton
    Left = 8
    Top = 8
    Width = 75
    Height = 25
    Caption = 'Find Ports'
    TabOrder = 1
    OnClick = FindPortBtnClick
  end
  object EnumNumSpin: TSpinEdit
    Left = 8
    Top = 48
    Width = 41
    Height = 22
    MaxValue = 15
    MinValue = 0
    TabOrder = 2
    Value = 0
    OnChange = EnumNumSpinChange
  end
  object OpenBtn: TButton
    Left = 56
    Top = 48
    Width = 81
    Height = 25
    Caption = 'Open Enum'
    TabOrder = 3
    OnClick = OpenBtnClick
  end
  object CloseBtn: TButton
    Left = 56
    Top = 80
    Width = 81
    Height = 25
    Caption = 'Close Enum'
    TabOrder = 4
    OnClick = CloseBtnClick
  end
  object MonoInfoBtn: TButton
    Left = 56
    Top = 112
    Width = 81
    Height = 25
    Caption = 'Get Mono Info'
    TabOrder = 5
    OnClick = MonoInfoBtnClick
  end
  object DialogList: TListBox
    Left = 232
    Top = 0
    Width = 233
    Height = 409
    ItemHeight = 13
    TabOrder = 6
  end
  object MonoIntLedChk: TCheckBox
    Left = 472
    Top = 8
    Width = 97
    Height = 17
    Caption = 'Int Led State'
    TabOrder = 7
    OnClick = MonoIntLedChkClick
  end
  object MoveStepsEdt: TEdit
    Left = 552
    Top = 32
    Width = 41
    Height = 21
    TabOrder = 8
  end
  object MoveStepsBtn: TButton
    Left = 600
    Top = 31
    Width = 75
    Height = 25
    Caption = 'Move Steps'
    TabOrder = 9
    OnClick = MoveStepsBtnClick
  end
  object RestoreFactoryBtn: TButton
    Left = 56
    Top = 256
    Width = 81
    Height = 25
    Caption = 'Restore Factory'
    TabOrder = 10
    OnClick = RestoreFactoryBtnClick
  end
  object ResetMonoBtn: TButton
    Left = 56
    Top = 288
    Width = 81
    Height = 25
    Caption = 'Reset Mono'
    TabOrder = 11
    OnClick = ResetMonoBtnClick
  end
  object JogBtn: TButton
    Left = 672
    Top = 80
    Width = 75
    Height = 25
    Caption = 'Jog Mono'
    TabOrder = 12
    TabStop = False
    OnMouseDown = JogBtnMouseDown
    OnMouseUp = JogBtnMouseUp
  end
  object DirRadio: TRadioGroup
    Left = 472
    Top = 56
    Width = 81
    Height = 49
    ItemIndex = 0
    Items.Strings = (
      'Up (nm)'
      'Down (nm)')
    TabOrder = 13
  end
  object JogRateRadio: TRadioGroup
    Left = 560
    Top = 56
    Width = 105
    Height = 49
    ItemIndex = 1
    Items.Strings = (
      'Max Jog Rate'
      'Scan Rate')
    TabOrder = 14
  end
  object ScanRateEdt: TEdit
    Left = 576
    Top = 112
    Width = 73
    Height = 21
    TabOrder = 15
  end
  object SetScanRateBtn: TButton
    Left = 656
    Top = 110
    Width = 89
    Height = 25
    Caption = 'Set Scan Rate'
    TabOrder = 16
    OnClick = SetScanRateBtnClick
  end
  object StartScanEdt: TEdit
    Left = 576
    Top = 136
    Width = 73
    Height = 21
    TabOrder = 17
  end
  object StopScanEdt: TEdit
    Left = 576
    Top = 160
    Width = 73
    Height = 21
    TabOrder = 18
  end
  object ScanBtn: TButton
    Left = 656
    Top = 160
    Width = 89
    Height = 25
    Caption = 'Scan'
    TabOrder = 19
    TabStop = False
    OnClick = ScanBtnClick
  end
  object MonoInitGratEdt: TEdit
    Left = 56
    Top = 152
    Width = 57
    Height = 21
    TabOrder = 20
  end
  object MonoInitWaveEdt: TEdit
    Left = 56
    Top = 184
    Width = 57
    Height = 21
    TabOrder = 21
  end
  object MonoInitScanEdt: TEdit
    Left = 56
    Top = 216
    Width = 57
    Height = 21
    TabOrder = 22
  end
  object MonoInitGratBtn: TButton
    Left = 120
    Top = 152
    Width = 105
    Height = 25
    Caption = 'Set Init Grating'
    TabOrder = 23
    OnClick = MonoInitGratBtnClick
  end
  object MonoInitWaveBtn: TButton
    Left = 120
    Top = 184
    Width = 105
    Height = 25
    Caption = 'Set Init Wave'
    TabOrder = 24
    OnClick = MonoInitWaveBtnClick
  end
  object MonoInitScanBtn: TButton
    Left = 120
    Top = 216
    Width = 105
    Height = 25
    Caption = 'Set Init Scan Rate'
    TabOrder = 25
    OnClick = MonoInitScanBtnClick
  end
  object UnInstallGratButton: TButton
    Left = 510
    Top = 288
    Width = 97
    Height = 25
    Caption = 'Uninstall Grating'
    TabOrder = 26
    OnClick = UnInstallGratButtonClick
  end
  object InstallGratButton: TButton
    Left = 614
    Top = 288
    Width = 97
    Height = 25
    Caption = 'Install Grating'
    TabOrder = 27
    OnClick = InstallGratButtonClick
  end
  object GratSpin: TSpinEdit
    Left = 536
    Top = 208
    Width = 33
    Height = 22
    MaxValue = 9
    MinValue = 1
    TabOrder = 28
    Value = 1
  end
  object GroovesEdit: TEdit
    Left = 550
    Top = 232
    Width = 67
    Height = 21
    TabOrder = 29
  end
  object BlazeEdit: TEdit
    Left = 550
    Top = 256
    Width = 67
    Height = 21
    TabOrder = 30
  end
  object BlazeGroup: TRadioGroup
    Left = 630
    Top = 208
    Width = 89
    Height = 70
    Items.Strings = (
      'mirror'
      'nm Blaze'
      'um Blaze'
      'HOL Blaze')
    TabOrder = 31
  end
  object DisplayWaveEdit: TEdit
    Left = 472
    Top = 336
    Width = 49
    Height = 21
    TabOrder = 32
  end
  object RefWaveEdit: TEdit
    Left = 472
    Top = 376
    Width = 49
    Height = 21
    TabOrder = 33
  end
  object CalcOffSetButton: TButton
    Left = 524
    Top = 336
    Width = 75
    Height = 25
    Caption = 'Calc Offset'
    TabOrder = 34
    OnClick = CalcOffSetButtonClick
  end
  object CalcGAdjButton: TButton
    Left = 524
    Top = 376
    Width = 75
    Height = 25
    Caption = 'Calc GAdjust'
    TabOrder = 35
    OnClick = CalcGAdjButtonClick
  end
  object SetOffsetButton: TButton
    Left = 672
    Top = 336
    Width = 75
    Height = 25
    Caption = 'Set Offset'
    TabOrder = 36
    OnClick = SetOffsetButtonClick
  end
  object SetGAdjButton: TButton
    Left = 672
    Top = 376
    Width = 75
    Height = 25
    Caption = 'Set GAdjust'
    TabOrder = 37
    OnClick = SetGAdjButtonClick
  end
end
