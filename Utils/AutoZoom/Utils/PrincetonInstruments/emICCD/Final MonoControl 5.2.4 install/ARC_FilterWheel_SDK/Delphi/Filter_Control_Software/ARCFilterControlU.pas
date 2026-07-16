unit ARCFilterControlU;

interface

uses
  Windows, Messages, SysUtils, Variants, Classes, Graphics, Controls, Forms,
  Dialogs,ARC_FilterWheelU, StdCtrls, Menus, ExtCtrls;

type
  TFilterWheelF = class(TForm)
    VerLabel: TLabel;
    MainMenu1: TMainMenu;
    File1: TMenuItem;
    Exit1: TMenuItem;
    Panel1: TPanel;
    GotoBtn: TButton;
    HomeBtn: TButton;
    PosEdt: TEdit;
    MaxFilterLabel: TLabel;
    MinFilterLabel: TLabel;
    ModelLabel: TLabel;
    SerialLabel: TLabel;
    FilterPosLabel: TLabel;
    Image1: TImage;
    Label1: TLabel;
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure Exit1Click(Sender: TObject);
    procedure GotoBtnClick(Sender: TObject);
    procedure HomeBtnClick(Sender: TObject);
  private
    Filter_Enum : integer;
    { Private declarations }
    procedure UpdateForm;
    procedure DisableForm;
    procedure EnableForm;
  public
    { Public declarations }
  end;

var
  FilterWheelF: TFilterWheelF;
implementation

{$R *.dfm}



procedure TFilterWheelF.FormCreate(Sender: TObject);
var
model_loop : integer;
model_str  : string;
Num_Filter : integer; 
begin
top  := 0;
left := 0;

if not Dynamic_Load_ARC_FilterWheel_DLL
then begin
     ModelLabel.caption := 'No Filter';
     SerialLabel.caption := '----';
     MinFilterLabel.caption := 'Min Filter = 0';
     MaxFilterLabel.caption := 'Min Filter = 0';
     ModelLabel.Enabled     := false;
     SerialLabel.Enabled    := false;
     MinFilterLabel.Enabled := false;
     MaxFilterLabel.Enabled := false;
     FilterPosLabel.Enabled := false;
     PosEdt.Enabled := false;
     GotoBtn.Enabled := false;
     HomeBtn.Enabled := false;
     VerLabel.caption := 'ARC_FilterWheel.dll Ver, ?.?.?';
     showmessage('Error : Failed to load ARC_FilterWheel.dll');
     close;
     end
else begin
     // setup filter wheel
     screen.cursor := crHourGlass;
     try
     Filter_Enum := -1; // have no filters setup
     if ARC_Search_For_Filter(Num_Filter)
     then begin
          for model_loop := 0 to Num_Filter - 1
           do begin
              if ARC_get_Filter_preOpen_Model(model_loop,model_str)
              then begin
                   // check to make sure we do not yet have a filterwheel setup
                   if not ARC_Valid_Filter_Enum(Filter_Enum)
                   then begin
                        // open the filter wheel
                        ARC_Open_Filter(model_loop,Filter_Enum);
                        end;
                   end;
              end;
          end;
     UpdateForm;
     finally
        screen.cursor := crDefault;
        end;
     end;
end;

procedure TFilterWheelF.FormDestroy(Sender: TObject);
begin
// unload filter wheel
try
if ARC_Valid_Filter_Enum(Filter_Enum)
   then ARC_Close_Filter(Filter_Enum);
except
   end;
end;

procedure TFilterWheelF.UpdateForm;
var
Major : integer;
Minor : integer;
Build : integer;
TempStr : string;
TempInt : integer;
begin
// Display DLL version
if ARC_Ver(Major,Minor,Build)
   then VerLabel.caption := 'ARC_FilterWheel.dll Ver, '
                            +IntToStr(Major)+'.'
                            +IntToStr(Minor)+'.'
                            +IntToStr(Build)
   else VerLabel.caption := 'ARC_FilterWheel.dll Ver, ?.?.?';
if ARC_Valid_Filter_Enum(Filter_Enum)
then begin
     // Display Model
     if ARC_get_Filter_Model(Filter_Enum,TempStr)
     then begin
          ModelLabel.caption := 'Model : ' +TempStr;
          end
     else begin
          ModelLabel.caption := 'Model : Error unable to read';
          end;
     // Display Serial Number
     if ARC_get_Filter_Serial(Filter_Enum,TempStr)
     then begin
          if trim(TempStr) <> ''
          then begin
               SerialLabel.Caption := 'Serial Number : ' +trim(TempStr);
               end
          else SerialLabel.Caption := '';
          end
     else SerialLabel.Caption := '';
     // Display Min FilterWheel
     if ARC_get_Filter_Min_Pos(Filter_Enum,TempInt)
        then MinFilterLabel.Caption := 'Min Filter = ' +IntToStr(TempInt)
        else MinFilterLabel.Caption := '';
     // Display Max FilterWheel
     if ARC_get_Filter_Max_Pos(Filter_Enum,TempInt)
        then MaxFilterLabel.Caption := 'Max Filter = ' +IntToStr(TempInt)
        else MaxFilterLabel.Caption := '';
     // Display the Current Filter
     if ARC_get_Filter_Position(Filter_Enum,TempInt)
        then PosEdt.Text := IntToStr(TempInt)
        else PosEdt.Text := '';
     end
else begin
     ModelLabel.Caption := 'FilterWheel Not Found !';
     ModelLabel.Font.Color  := clRed;
     SerialLabel.visible    := false;
     MinFilterLabel.visible := false;
     MaxFilterLabel.visible := false;
     DisableForm;
     end;
end;

procedure TFilterWheelF.DisableForm;
begin
PosEdt.enabled := false;
GotoBtn.enabled := false;
HomeBtn.enabled := false;
end;

procedure TFilterWheelF.EnableForm;
begin
PosEdt.enabled  := true;
GotoBtn.enabled := true;
HomeBtn.enabled := true;
end;

procedure TFilterWheelF.Exit1Click(Sender: TObject);
begin
close;
end;

procedure TFilterWheelF.GotoBtnClick(Sender: TObject);
var
NewFilter : integer;
begin
try
Screen.Cursor := crHourGlass;
DisableForm;
NewFilter := Round(StrToFloat(Trim(PosEdt.text)));
ARC_set_Filter_Position(Filter_Enum,NewFilter);
UpdateForm;
Screen.Cursor := crDefault;
except
   Screen.Cursor := crDefault;
   showmessage('Error : Mal-Formed Position');
   end;
// re enable the form
EnableForm;
end;

procedure TFilterWheelF.HomeBtnClick(Sender: TObject);
begin
try
screen.cursor := crHourGlass;
DisableForm;
ARC_Filter_Home(Filter_Enum);
finally
   screen.cursor := crDefault;
   UpdateForm;
   EnableForm;
   end;
end;

end.
