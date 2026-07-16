unit Filter_DLL_TestU;

interface

uses
  Windows, Messages, SysUtils, Variants, Classes, Graphics, Controls, Forms,
  Dialogs , ARC_FilterWheelU, StdCtrls, Spin;

type
  TFilterTestForm = class(TForm)
    SendCMDLabel: TLabel;
    Label1: TLabel;
    ConnectedChk: TCheckBox;
    FindPortBtn: TButton;
    EnumNumSpin: TSpinEdit;
    OpenBtn: TButton;
    CloseBtn: TButton;
    SendCMDEdt: TEdit;
    SendCMDBtn: TButton;
    FilterInfoBtn: TButton;
    DialogList: TListBox;
    NewFilterEdt: TEdit;
    SetFilterBtn: TButton;
    HomeFilterBtn: TButton;
    procedure FindPortBtnClick(Sender: TObject);
    procedure EnumNumSpinChange(Sender: TObject);
    procedure OpenBtnClick(Sender: TObject);
    procedure CloseBtnClick(Sender: TObject);
    procedure FilterInfoBtnClick(Sender: TObject);
    procedure SendCMDBtnClick(Sender: TObject);
    procedure SetFilterBtnClick(Sender: TObject);
    procedure HomeFilterBtnClick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  FilterTestForm: TFilterTestForm;

implementation

{$R *.dfm}

procedure TFilterTestForm.FindPortBtnClick(Sender: TObject);
var
Num_Filter : integer;
model_loop : integer;
model_str  : string;
begin
screen.cursor := crHourGlass;
try
DialogList.Items.clear;

if ARC_Search_For_Filter(Num_Filter)
then begin
     for model_loop := 0 to Num_Filter - 1
      do begin
         if ARC_get_Filter_preOpen_Model(model_loop,model_str)
            then DialogList.Items.Add('Enum : ' +IntToStr(model_loop) +' ' +model_str);
         end;
     end
else begin
     DialogList.Items.Add('No Filters attached');
     end;
finally
   screen.cursor := crDefault;
   end;
end;

procedure TFilterTestForm.EnumNumSpinChange(Sender: TObject);
begin
try
// indicate if the current enum is open or not
ConnectedChk.checked := ARC_Valid_Filter_Enum(EnumNumSpin.Value);
except
   ConnectedChk.checked := false;
   end;
end;

procedure TFilterTestForm.OpenBtnClick(Sender: TObject);
var
dummy : integer;
begin
screen.cursor := crHourGlass;
try
// open the mono port
if not ARC_Open_Filter(EnumNumSpin.Value,dummy)
       then showmessage('Error : Failed to Open Port');
// update if the port is or is not opened
EnumNumSpinChange(Sender);
finally
   screen.cursor := crDefault;
   end;
end;

procedure TFilterTestForm.CloseBtnClick(Sender: TObject);
begin
// close the mono port
if not ARC_Close_Filter(EnumNumSpin.Value)
       then showmessage('Error : Failed to Close Port');
// update if the port is or is not opened
EnumNumSpinChange(Sender);
end;

procedure TFilterTestForm.FilterInfoBtnClick(Sender: TObject);
var
outstr : string;
outint : integer;
begin
DialogList.Items.Clear;
if ARC_get_Filter_Model(EnumNumSpin.Value,outstr)
then begin
     DialogList.Items.add('Model : ' +outstr);
     end
else DialogList.Items.add('Error : No Model');
if ARC_get_Filter_Serial(EnumNumSpin.Value,outstr)
then begin
     DialogList.Items.add('Serial : ' +outstr);
     end
else DialogList.Items.add('Error : No Serial');

// get filter info
if ARC_get_Filter_Present(EnumNumSpin.Value)
then begin
     if ARC_get_Filter_Position(EnumNumSpin.Value,outint)
        then DialogList.Items.add('Filter Position : ' +IntToStr(outInt))
        else DialogList.Items.add('Error : Filter Position');
     if ARC_get_Filter_Min_Pos(EnumNumSpin.Value,outint)
        then DialogList.Items.add('Min Filter Position : ' +IntToStr(outInt))
        else DialogList.Items.add('Error : Min Filter Position');
     if ARC_get_Filter_Max_Pos(EnumNumSpin.Value,outint)
        then DialogList.Items.add('Max Filter Position : ' +IntToStr(outInt))
        else DialogList.Items.add('Error : Max Filter Position');
     end
else begin
     DialogList.Items.add('No Filter');
     end;
end;

procedure TFilterTestForm.SendCMDBtnClick(Sender: TObject);
var
ReturnStr : string;
begin
screen.cursor := crHourGlass;
try
// send a command to the instrument, note the function adds the required CR (^M)
// to the function, so the user does not need to. Placing a CR in the command
// string can have unpredictable results.
if not ARC_Send_CMD_To_Filter(EnumNumSpin.Value,SendCMDEdt.text,ReturnStr,60000)
   then showmessage('Error : Failed to Send Command');
SendCMDLabel.Caption := 'Returned : ' +ReturnStr;
except
   end;
screen.cursor := crDefault;
end;

procedure TFilterTestForm.SetFilterBtnClick(Sender: TObject);
begin
screen.cursor := crHourGlass;
try
if not ARC_set_Filter_Position(EnumNumSpin.Value,round(StrToFloat(Trim(NewFilterEdt.Text))))
   then showmessage('Error : Failed to set Filter Position');
except
   end;
screen.cursor := crDefault;
end;

procedure TFilterTestForm.HomeFilterBtnClick(Sender: TObject);
begin
screen.cursor := crHourGlass;
try
if not ARC_Filter_Home(EnumNumSpin.Value)
   then showmessage('Error : Failed to Home Filter');
except
   end;
screen.cursor := crDefault;
end;

procedure TFilterTestForm.FormCreate(Sender: TObject);
begin
top  := 0;
left := 0;

if Dynamic_Load_ARC_FilterWheel_DLL
   then DialogList.Items.Add('Loaded ARC_FilterWheel_Pro.dll')
   else DialogList.Items.Add('Error : Failed to load ARC_FilterWheel_Pro.dll');
end;

procedure TFilterTestForm.FormDestroy(Sender: TObject);
begin
unload_ARC_FilterWheel_DLL;
end;

end.
