unit Direct_ComU;

interface

uses
  Windows,
  Messages,
  SysUtils,
  Variants,
  Classes,
  Graphics,
  Controls,
  Forms,
  Dialogs,
  StdCtrls,
  Spin,
  ARC_SpectraProU;

type
  TDirect_ComF = class(TForm)
    Button1: TButton;
    ModelLabel: TLabel;
    WaveLabel: TLabel;
    EnumLabel: TLabel;
    SpinEdit1: TSpinEdit;
    Label4: TLabel;
    procedure SpinEdit1Change(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure Button1Click(Sender: TObject);
  private
    { Private declarations }
    Mono_Enum  : integer;
    Com_Num    : integer;
  public
    { Public declarations }
  end;

var
  Direct_ComF: TDirect_ComF;

implementation

{$R *.dfm}

procedure TDirect_ComF.SpinEdit1Change(Sender: TObject);
begin
Try
Com_Num := SpinEdit1.Value;
except
   end;
end;

procedure TDirect_ComF.FormCreate(Sender: TObject);
begin
top  := 0;
left := 0;

Com_Num := 2; // provide an initial value for Com_Num

if not Dynamic_Load_ARC_SpectraPro_DLL
then begin
     showmessage('Error : Failed to load ARC_Spectra_Pro.dll');
     close;
     end;
end;

procedure TDirect_ComF.Button1Click(Sender: TObject);
var
outstr  : string;
outdbl  : double;
begin
screen.cursor := crhourglass;
try
if ARC_Open_Mono_Port(Com_Num,Mono_Enum)
then begin
     // example of some functions that use the dll
     ARC_get_Mono_Model(Mono_Enum,outstr);
     ModelLabel.caption := 'Model : ' +outstr;
     ARC_get_Mono_Wavelength_nm(Mono_Enum,outdbl);
     WaveLabel.caption  := 'Wave : ' +format('%.2f',[outdbl])  +' nm';
     EnumLabel.caption  := 'Enum : ' +IntToStr(Mono_Enum);
     // we are done with the port for this example code, close it
     // normally you will open the port in one function
     // have other functions for working witht he instrument
     // and close the instrument as you exit the software
     ARC_Close_Mono(Mono_Enum);
     end
else begin
     ModelLabel.caption := 'Model :';
     WaveLabel.caption  := 'Wave : --- nm';
     EnumLabel.caption  := 'Enum :';
     end;
finally
   screen.Cursor := crdefault;
   end;
end;

end.
