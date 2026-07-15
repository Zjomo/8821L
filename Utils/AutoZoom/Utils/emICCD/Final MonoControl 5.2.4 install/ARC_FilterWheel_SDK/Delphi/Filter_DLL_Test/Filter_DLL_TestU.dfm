object FilterTestForm: TFilterTestForm
  Left = 191
  Top = 60
  Width = 477
  Height = 345
  HorzScrollBar.Visible = False
  VertScrollBar.Visible = False
  Caption = 'FilterTestForm'
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
    Top = 280
    Width = 75
    Height = 13
    Caption = 'SendCMDLabel'
  end
  object Label1: TLabel
    Left = 0
    Top = 232
    Width = 130
    Height = 13
    Caption = 'Send Raw Mono Command'
  end
  object ConnectedChk: TCheckBox
    Left = 152
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
    Top = 252
    Width = 465
    Height = 21
    TabOrder = 5
    Text = 'SendCMDEdt'
  end
  object SendCMDBtn: TButton
    Left = 0
    Top = 280
    Width = 75
    Height = 25
    Caption = 'Send CMD'
    TabOrder = 6
    OnClick = SendCMDBtnClick
  end
  object FilterInfoBtn: TButton
    Left = 64
    Top = 112
    Width = 81
    Height = 25
    Caption = 'Get Filter Info'
    TabOrder = 7
    OnClick = FilterInfoBtnClick
  end
  object DialogList: TListBox
    Left = 232
    Top = 0
    Width = 233
    Height = 249
    ItemHeight = 13
    TabOrder = 8
  end
  object NewFilterEdt: TEdit
    Left = 8
    Top = 168
    Width = 41
    Height = 21
    TabOrder = 9
    Text = 'Filter'
  end
  object SetFilterBtn: TButton
    Left = 56
    Top = 168
    Width = 75
    Height = 25
    Caption = 'Set Filter'
    TabOrder = 10
    OnClick = SetFilterBtnClick
  end
  object HomeFilterBtn: TButton
    Left = 136
    Top = 168
    Width = 75
    Height = 25
    Caption = 'Home Filter'
    TabOrder = 11
    OnClick = HomeFilterBtnClick
  end
end
