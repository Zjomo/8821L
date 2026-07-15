program Direct_Com;

uses
  Forms,
  Direct_ComU in 'Direct_ComU.pas' {Direct_ComF};

{$R *.res}

begin
  Application.Initialize;
  Application.CreateForm(TDirect_ComF, Direct_ComF);
  Application.Run;
end.
