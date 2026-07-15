object TestForm: TTestForm
  Left = 189
  Top = 78
  Width = 760
  Height = 347
  HorzScrollBar.Visible = False
  VertScrollBar.Visible = False
  Caption = 'TestForm'
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
  object SendCMDLabel: TLabel
    Left = 80
    Top = 240
    Width = 75
    Height = 13
    Caption = 'SendCMDLabel'
  end
  object Label1: TLabel
    Left = 0
    Top = 192
    Width = 130
    Height = 13
    Caption = 'Send Raw Mono Command'
  end
  object CenterWaveLabel: TLabel
    Left = 696
    Top = 132
    Width = 48
    Height = 13
    Caption = 'Center nm'
  end
  object NewWaveLabel: TLabel
    Left = 468
    Top = 56
    Width = 54
    Height = 13
    Caption = 'New Wave'
  end
  object ConnectedChk: TCheckBox
    Left = 152
    Top = 56
    Width = 97
    Height = 17
    Caption = 'Connected'
    TabOrder = 5
  end
  object FindPortBtn: TButton
    Left = 8
    Top = 8
    Width = 75
    Height = 25
    Caption = 'Find Ports'
    TabOrder = 0
    OnClick = FindPortBtnClick
  end
  object EnumNumSpin: TSpinEdit
    Left = 8
    Top = 48
    Width = 49
    Height = 22
    MaxValue = 15
    MinValue = 0
    TabOrder = 2
    Value = 0
    OnChange = EnumNumSpinChange
  end
  object OpenBtn: TButton
    Left = 64
    Top = 48
    Width = 81
    Height = 25
    Caption = 'Open Enum'
    TabOrder = 3
    OnClick = OpenBtnClick
  end
  object CloseBtn: TButton
    Left = 64
    Top = 80
    Width = 81
    Height = 25
    Caption = 'Close Enum'
    TabOrder = 4
    OnClick = CloseBtnClick
  end
  object SendCMDEdt: TEdit
    Left = 0
    Top = 212
    Width = 233
    Height = 21
    TabOrder = 6
    Text = 'SendCMDEdt'
  end
  object SendCMDBtn: TButton
    Left = 0
    Top = 240
    Width = 75
    Height = 25
    Caption = 'Send CMD'
    TabOrder = 7
    OnClick = SendCMDBtnClick
  end
  object MonoInfoBtn: TButton
    Left = 64
    Top = 112
    Width = 81
    Height = 25
    Caption = 'Get Mono Info'
    TabOrder = 8
    OnClick = MonoInfoBtnClick
  end
  object DialogList: TListBox
    Left = 232
    Top = 0
    Width = 233
    Height = 313
    ItemHeight = 13
    TabOrder = 1
  end
  object SetWaveEdt: TEdit
    Left = 468
    Top = 72
    Width = 57
    Height = 21
    TabOrder = 9
  end
  object WaveUnitRadio: TRadioGroup
    Left = 528
    Top = 0
    Width = 89
    Height = 153
    Items.Strings = (
      'Ang'
      'nm'
      'microns'
      'eV'
      'Abs CM -1'
      'Rel CM - 1')
    TabOrder = 10
  end
  object CenterWaveEdt: TEdit
    Left = 624
    Top = 128
    Width = 65
    Height = 21
    TabOrder = 11
  end
  object SetWaveBtn: TButton
    Left = 528
    Top = 160
    Width = 89
    Height = 25
    Caption = 'Set Wavelength'
    TabOrder = 12
    OnClick = SetWaveBtnClick
  end
  object TurretSpin: TSpinEdit
    Left = 480
    Top = 192
    Width = 41
    Height = 22
    MaxValue = 4
    MinValue = 1
    TabOrder = 13
    Value = 1
  end
  object GratingSpin: TSpinEdit
    Left = 480
    Top = 224
    Width = 41
    Height = 22
    MaxValue = 15
    MinValue = 1
    TabOrder = 14
    Value = 1
  end
  object SetTurretBtn: TButton
    Left = 528
    Top = 192
    Width = 89
    Height = 25
    Caption = 'Set Turret'
    TabOrder = 15
    OnClick = SetTurretBtnClick
  end
  object SetGratingBtn: TButton
    Left = 528
    Top = 224
    Width = 89
    Height = 25
    Caption = 'Set Grating'
    TabOrder = 16
    OnClick = SetGratingBtnClick
  end
  object SetSlitWidthBtn: TButton
    Left = 576
    Top = 256
    Width = 75
    Height = 25
    Caption = 'Set Slit Width'
    TabOrder = 17
    OnClick = SetSlitWidthBtnClick
  end
  object HomeSlitBtn: TButton
    Left = 656
    Top = 256
    Width = 75
    Height = 25
    Caption = 'Home Slit'
    TabOrder = 18
    OnClick = HomeSlitBtnClick
  end
  object NewSlitWidthEdt: TEdit
    Left = 528
    Top = 256
    Width = 41
    Height = 21
    TabOrder = 19
    Text = 'Slit'
  end
  object NewFilterEdt: TEdit
    Left = 528
    Top = 288
    Width = 41
    Height = 21
    TabOrder = 20
    Text = 'Filter'
  end
  object SetFilterBtn: TButton
    Left = 576
    Top = 288
    Width = 75
    Height = 25
    Caption = 'Set Filter'
    TabOrder = 21
    OnClick = SetFilterBtnClick
  end
  object HomeFilterBtn: TButton
    Left = 656
    Top = 288
    Width = 75
    Height = 25
    Caption = 'Home Filter'
    TabOrder = 22
    OnClick = HomeFilterBtnClick
  end
  object SlitSpin: TSpinEdit
    Left = 480
    Top = 256
    Width = 41
    Height = 22
    MaxValue = 8
    MinValue = 1
    TabOrder = 23
    Value = 1
  end
end
