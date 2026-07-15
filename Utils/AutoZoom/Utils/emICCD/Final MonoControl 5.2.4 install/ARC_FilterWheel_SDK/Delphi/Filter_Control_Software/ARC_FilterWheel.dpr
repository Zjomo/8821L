program ARC_FilterWheel;

uses
  Forms,
  ARCFilterControlU in 'ARCFilterControlU.pas' {FilterWheelF};

{$R *.res}

begin
  Application.Initialize;
  Application.CreateForm(TFilterWheelF, FilterWheelF);
  Application.Run;
end.
