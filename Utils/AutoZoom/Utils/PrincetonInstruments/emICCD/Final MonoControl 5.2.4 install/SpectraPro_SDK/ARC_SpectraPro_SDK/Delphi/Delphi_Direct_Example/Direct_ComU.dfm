object Direct_ComF: TDirect_ComF
  Left = 191
  Top = 3
  Width = 230
  Height = 179
  Caption = 'Direct Com Port Connect'
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'MS Sans Serif'
  Font.Style = []
  OldCreateOrder = False
  OnCreate = FormCreate
  PixelsPerInch = 96
  TextHeight = 13
  object ModelLabel: TLabel
    Left = 72
    Top = 80
    Width = 35
    Height = 13
    Caption = 'Model :'
  end
  object WaveLabel: TLabel
    Left = 72
    Top = 104
    Width = 64
    Height = 13
    Caption = 'Wave : --- nm'
  end
  object EnumLabel: TLabel
    Left = 72
    Top = 128
    Width = 36
    Height = 13
    Caption = 'Enum : '
  end
  object Label4: TLabel
    Left = 56
    Top = 16
    Width = 49
    Height = 13
    Caption = 'Com Port :'
  end
  object Button1: TButton
    Left = 72
    Top = 40
    Width = 75
    Height = 25
    Caption = 'Open Port'
    TabOrder = 0
    OnClick = Button1Click
  end
  object SpinEdit1: TSpinEdit
    Left = 112
    Top = 12
    Width = 41
    Height = 22
    MaxValue = 0
    MinValue = 0
    TabOrder = 1
    Value = 2
    OnChange = SpinEdit1Change
  end
end
