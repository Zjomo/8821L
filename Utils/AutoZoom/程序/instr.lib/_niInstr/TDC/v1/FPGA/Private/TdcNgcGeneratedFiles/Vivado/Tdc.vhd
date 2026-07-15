-- Copyright 1986-2014 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2014.4_AR64601_AR63880_AR63479_AR62969_(AR63524_AR64594) (win64) Build 0 Tue May 19 17:22:27 MDT
--               2015
-- Date        : Wed Jun 24 13:40:43 2015
-- Host        : running 64-bit Service Pack 1  (build 7601)
-- Command     : write_vhdl -mode funcsim {C:\NIFPGA\iptemp\ipinF00040DCBCC74A4682E9AB62BDA02F6F\Vivado\Tdc.vhd}
-- Design      : Tdc
-- Purpose     : This VHDL netlist is a functional simulation representation of the design and should not be modified or
--               synthesized. This netlist cannot be used for SDF annotated simulation.
-- Device      : xc7k410tfbg900-2
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity DFlopSlvResetVal is
  port (
    Clk : in STD_LOGIC;
    cQ : out STD_LOGIC_VECTOR ( 10 downto 0 );
    cD : in STD_LOGIC_VECTOR ( 10 downto 0 );
    aReset : in STD_LOGIC_VECTOR ( 0 to 0 );
    cEn : in STD_LOGIC_VECTOR ( 0 to 0 )
  );
end DFlopSlvResetVal;

architecture STRUCTURE of DFlopSlvResetVal is
  signal cLclQ : STD_LOGIC_VECTOR ( 10 downto 0 );
  attribute XSTLIB : boolean;
  attribute XSTLIB of cLclQ_0 : label is std.standard.true;
  attribute XSTLIB of cLclQ_1 : label is std.standard.true;
  attribute XSTLIB of cLclQ_10 : label is std.standard.true;
  attribute XSTLIB of cLclQ_2 : label is std.standard.true;
  attribute XSTLIB of cLclQ_3 : label is std.standard.true;
  attribute XSTLIB of cLclQ_4 : label is std.standard.true;
  attribute XSTLIB of cLclQ_5 : label is std.standard.true;
  attribute XSTLIB of cLclQ_6 : label is std.standard.true;
  attribute XSTLIB of cLclQ_7 : label is std.standard.true;
  attribute XSTLIB of cLclQ_8 : label is std.standard.true;
  attribute XSTLIB of cLclQ_9 : label is std.standard.true;
begin
  cQ(10 downto 0) <= cLclQ(10 downto 0);
cLclQ_0: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => Clk,
      CE => cEn(0),
      CLR => aReset(0),
      D => cD(0),
      Q => cLclQ(0)
    );
cLclQ_1: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => Clk,
      CE => cEn(0),
      CLR => aReset(0),
      D => cD(1),
      Q => cLclQ(1)
    );
cLclQ_10: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => Clk,
      CE => cEn(0),
      CLR => aReset(0),
      D => cD(10),
      Q => cLclQ(10)
    );
cLclQ_2: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => Clk,
      CE => cEn(0),
      CLR => aReset(0),
      D => cD(2),
      Q => cLclQ(2)
    );
cLclQ_3: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => Clk,
      CE => cEn(0),
      CLR => aReset(0),
      D => cD(3),
      Q => cLclQ(3)
    );
cLclQ_4: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => Clk,
      CE => cEn(0),
      CLR => aReset(0),
      D => cD(4),
      Q => cLclQ(4)
    );
cLclQ_5: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => Clk,
      CE => cEn(0),
      CLR => aReset(0),
      D => cD(5),
      Q => cLclQ(5)
    );
cLclQ_6: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => Clk,
      CE => cEn(0),
      CLR => aReset(0),
      D => cD(6),
      Q => cLclQ(6)
    );
cLclQ_7: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => Clk,
      CE => cEn(0),
      CLR => aReset(0),
      D => cD(7),
      Q => cLclQ(7)
    );
cLclQ_8: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => Clk,
      CE => cEn(0),
      CLR => aReset(0),
      D => cD(8),
      Q => cLclQ(8)
    );
cLclQ_9: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => Clk,
      CE => cEn(0),
      CLR => aReset(0),
      D => cD(9),
      Q => cLclQ(9)
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity HandshakeBase is
  port (
    OClk : in STD_LOGIC;
    IClk : in STD_LOGIC;
    iStoredData : out STD_LOGIC_VECTOR ( 10 downto 0 );
    oDataValid : out STD_LOGIC_VECTOR ( 0 to 0 );
    oData : out STD_LOGIC_VECTOR ( 10 downto 0 );
    iReady : out STD_LOGIC_VECTOR ( 0 to 0 );
    iData : in STD_LOGIC_VECTOR ( 10 downto 0 );
    aReset : in STD_LOGIC_VECTOR ( 0 to 0 );
    iPush : in STD_LOGIC_VECTOR ( 0 to 0 );
    oDataAck : in STD_LOGIC_VECTOR ( 0 to 0 )
  );
end HandshakeBase;

architecture STRUCTURE of HandshakeBase is
  signal iDlyPush : STD_LOGIC_VECTOR ( 0 to 0 );
  signal iLclReady : STD_LOGIC_VECTOR ( 0 to 0 );
  signal iLclReady_0_and0000 : STD_LOGIC;
  signal iLclStoredData : STD_LOGIC_VECTOR ( 10 downto 0 );
  signal iPushPulse : STD_LOGIC;
  signal iPushToggle : STD_LOGIC_VECTOR ( 0 to 0 );
  signal iPushToggle_0_not0001 : STD_LOGIC;
  signal iRdyPushToggle : STD_LOGIC_VECTOR ( 0 to 0 );
  signal iRdyPushToggle_ms : STD_LOGIC;
  signal iReset : STD_LOGIC_VECTOR ( 0 to 0 );
  signal iReset_ms : STD_LOGIC;
  signal oPushToggle0_ms : STD_LOGIC;
  signal oPushToggle1 : STD_LOGIC_VECTOR ( 0 to 0 );
  signal oPushToggle2 : STD_LOGIC_VECTOR ( 0 to 0 );
  signal oPushToggleChanged : STD_LOGIC;
  signal oPushToggleToReady : STD_LOGIC_VECTOR ( 0 to 0 );
  attribute BUS_INFO : string;
  attribute BUS_INFO of \BlkOut.ODataFlop\ : label is "1:INPUT:cEn(0:0)";
  attribute NLW_MACRO_ALIAS : string;
  attribute NLW_MACRO_ALIAS of \BlkOut.ODataFlop\ : label is "DFlopSlvResetVal_BlkOut.ODataFlop";
  attribute NLW_MACRO_TAG : integer;
  attribute NLW_MACRO_TAG of \BlkOut.ODataFlop\ : label is 3;
  attribute NLW_UNIQUE_ID : integer;
  attribute NLW_UNIQUE_ID of \BlkOut.ODataFlop\ : label is 0;
  attribute XSTLIB : boolean;
  attribute XSTLIB of \Mxor_oPushToggleChanged(0)_Result1\ : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM : string;
  attribute XILINX_LEGACY_PRIM of iDlyPush_0 : label is "FDC";
  attribute XSTLIB of iDlyPush_0 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of iLclReady_0 : label is "FDC";
  attribute XSTLIB of iLclReady_0 : label is std.standard.true;
  attribute XSTLIB of iLclReady_0_and00001 : label is std.standard.true;
  attribute XSTLIB of iLclStoredData_0 : label is std.standard.true;
  attribute XSTLIB of iLclStoredData_1 : label is std.standard.true;
  attribute XSTLIB of iLclStoredData_10 : label is std.standard.true;
  attribute XSTLIB of iLclStoredData_2 : label is std.standard.true;
  attribute XSTLIB of iLclStoredData_3 : label is std.standard.true;
  attribute XSTLIB of iLclStoredData_4 : label is std.standard.true;
  attribute XSTLIB of iLclStoredData_5 : label is std.standard.true;
  attribute XSTLIB of iLclStoredData_6 : label is std.standard.true;
  attribute XSTLIB of iLclStoredData_7 : label is std.standard.true;
  attribute XSTLIB of iLclStoredData_8 : label is std.standard.true;
  attribute XSTLIB of iLclStoredData_9 : label is std.standard.true;
  attribute XSTLIB of iPushPulse_0_and00001 : label is std.standard.true;
  attribute XSTLIB of iPushToggle_0 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of iPushToggle_0_not00011_INV_0 : label is "INV";
  attribute XSTLIB of iPushToggle_0_not00011_INV_0 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of iRdyPushToggle_0 : label is "FDC";
  attribute XSTLIB of iRdyPushToggle_0 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of iRdyPushToggle_ms_0 : label is "FDC";
  attribute XSTLIB of iRdyPushToggle_ms_0 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of iReset_0 : label is "FDP";
  attribute XSTLIB of iReset_0 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of iReset_ms_0 : label is "FDP";
  attribute XSTLIB of iReset_ms_0 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of oDataValid_0 : label is "FDC";
  attribute XSTLIB of oDataValid_0 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of oPushToggle0_ms_0 : label is "FDC";
  attribute XSTLIB of oPushToggle0_ms_0 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of oPushToggle1_0 : label is "FDC";
  attribute XSTLIB of oPushToggle1_0 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of oPushToggle2_0 : label is "FDC";
  attribute XSTLIB of oPushToggle2_0 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of oPushToggleToReady_0 : label is "FDC";
  attribute XSTLIB of oPushToggleToReady_0 : label is std.standard.true;
begin
  iReady(0) <= iLclReady(0);
  iStoredData(10 downto 0) <= iLclStoredData(10 downto 0);
\BlkOut.ODataFlop\: entity work.DFlopSlvResetVal
    port map (
      Clk => OClk,
      aReset(0) => aReset(0),
      cD(10 downto 0) => iLclStoredData(10 downto 0),
      cEn(0) => oPushToggleChanged,
      cQ(10 downto 0) => oData(10 downto 0)
    );
\Mxor_oPushToggleChanged(0)_Result1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
    port map (
      I0 => oPushToggle1(0),
      I1 => oPushToggle2(0),
      O => oPushToggleChanged
    );
iDlyPush_0: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => IClk,
      CE => '1',
      CLR => aReset(0),
      D => iPush(0),
      Q => iDlyPush(0)
    );
iLclReady_0: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => IClk,
      CE => '1',
      CLR => aReset(0),
      D => iLclReady_0_and0000,
      Q => iLclReady(0)
    );
iLclReady_0_and00001: unisim.vcomponents.LUT4
    generic map(
      INIT => X"1001"
    )
    port map (
      I0 => iPush(0),
      I1 => iReset(0),
      I2 => iRdyPushToggle(0),
      I3 => iPushToggle(0),
      O => iLclReady_0_and0000
    );
iLclStoredData_0: unisim.vcomponents.FDCE
    port map (
      C => IClk,
      CE => iPushPulse,
      CLR => aReset(0),
      D => iData(0),
      Q => iLclStoredData(0)
    );
iLclStoredData_1: unisim.vcomponents.FDCE
    port map (
      C => IClk,
      CE => iPushPulse,
      CLR => aReset(0),
      D => iData(1),
      Q => iLclStoredData(1)
    );
iLclStoredData_10: unisim.vcomponents.FDCE
    port map (
      C => IClk,
      CE => iPushPulse,
      CLR => aReset(0),
      D => iData(10),
      Q => iLclStoredData(10)
    );
iLclStoredData_2: unisim.vcomponents.FDCE
    port map (
      C => IClk,
      CE => iPushPulse,
      CLR => aReset(0),
      D => iData(2),
      Q => iLclStoredData(2)
    );
iLclStoredData_3: unisim.vcomponents.FDCE
    port map (
      C => IClk,
      CE => iPushPulse,
      CLR => aReset(0),
      D => iData(3),
      Q => iLclStoredData(3)
    );
iLclStoredData_4: unisim.vcomponents.FDCE
    port map (
      C => IClk,
      CE => iPushPulse,
      CLR => aReset(0),
      D => iData(4),
      Q => iLclStoredData(4)
    );
iLclStoredData_5: unisim.vcomponents.FDCE
    port map (
      C => IClk,
      CE => iPushPulse,
      CLR => aReset(0),
      D => iData(5),
      Q => iLclStoredData(5)
    );
iLclStoredData_6: unisim.vcomponents.FDCE
    port map (
      C => IClk,
      CE => iPushPulse,
      CLR => aReset(0),
      D => iData(6),
      Q => iLclStoredData(6)
    );
iLclStoredData_7: unisim.vcomponents.FDCE
    port map (
      C => IClk,
      CE => iPushPulse,
      CLR => aReset(0),
      D => iData(7),
      Q => iLclStoredData(7)
    );
iLclStoredData_8: unisim.vcomponents.FDCE
    port map (
      C => IClk,
      CE => iPushPulse,
      CLR => aReset(0),
      D => iData(8),
      Q => iLclStoredData(8)
    );
iLclStoredData_9: unisim.vcomponents.FDCE
    port map (
      C => IClk,
      CE => iPushPulse,
      CLR => aReset(0),
      D => iData(9),
      Q => iLclStoredData(9)
    );
iPushPulse_0_and00001: unisim.vcomponents.LUT2
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => iPush(0),
      I1 => iDlyPush(0),
      O => iPushPulse
    );
iPushToggle_0: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => IClk,
      CE => iPushPulse,
      CLR => aReset(0),
      D => iPushToggle_0_not0001,
      Q => iPushToggle(0)
    );
iPushToggle_0_not00011_INV_0: unisim.vcomponents.LUT1
    generic map(
      INIT => X"1"
    )
    port map (
      I0 => iPushToggle(0),
      O => iPushToggle_0_not0001
    );
iRdyPushToggle_0: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => IClk,
      CE => '1',
      CLR => aReset(0),
      D => iRdyPushToggle_ms,
      Q => iRdyPushToggle(0)
    );
iRdyPushToggle_ms_0: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => IClk,
      CE => '1',
      CLR => aReset(0),
      D => oPushToggleToReady(0),
      Q => iRdyPushToggle_ms
    );
iReset_0: unisim.vcomponents.FDPE
    generic map(
      INIT => '1'
    )
    port map (
      C => IClk,
      CE => '1',
      D => iReset_ms,
      PRE => aReset(0),
      Q => iReset(0)
    );
iReset_ms_0: unisim.vcomponents.FDPE
    generic map(
      INIT => '1'
    )
    port map (
      C => IClk,
      CE => '1',
      D => '0',
      PRE => aReset(0),
      Q => iReset_ms
    );
oDataValid_0: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => OClk,
      CE => '1',
      CLR => aReset(0),
      D => oPushToggleChanged,
      Q => oDataValid(0)
    );
oPushToggle0_ms_0: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => OClk,
      CE => '1',
      CLR => aReset(0),
      D => iPushToggle(0),
      Q => oPushToggle0_ms
    );
oPushToggle1_0: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => OClk,
      CE => '1',
      CLR => aReset(0),
      D => oPushToggle0_ms,
      Q => oPushToggle1(0)
    );
oPushToggle2_0: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => OClk,
      CE => '1',
      CLR => aReset(0),
      D => oPushToggle1(0),
      Q => oPushToggle2(0)
    );
oPushToggleToReady_0: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => OClk,
      CE => '1',
      CLR => aReset(0),
      D => oPushToggle2(0),
      Q => oPushToggleToReady(0)
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity HandshakeSLV is
  port (
    OClk : in STD_LOGIC;
    IClk : in STD_LOGIC;
    oDataValid : out STD_LOGIC_VECTOR ( 0 to 0 );
    oData : out STD_LOGIC_VECTOR ( 10 downto 0 );
    iReady : out STD_LOGIC_VECTOR ( 0 to 0 );
    iData : in STD_LOGIC_VECTOR ( 10 downto 0 );
    aReset : in STD_LOGIC_VECTOR ( 0 to 0 );
    iPush : in STD_LOGIC_VECTOR ( 0 to 0 )
  );
end HandshakeSLV;

architecture STRUCTURE of HandshakeSLV is
  signal NLW_HBx_iStoredData_UNCONNECTED : STD_LOGIC_VECTOR ( 10 downto 0 );
  signal NLW_HBx_oDataAck_UNCONNECTED : STD_LOGIC_VECTOR ( 0 to 0 );
  attribute BUS_INFO : string;
  attribute BUS_INFO of HBx : label is "1:INPUT:oDataAck(0:0)";
  attribute KEEP_HIERARCHY : string;
  attribute KEEP_HIERARCHY of HBx : label is "TRUE";
  attribute NLW_MACRO_ALIAS : string;
  attribute NLW_MACRO_ALIAS of HBx : label is "HandshakeBase_HBx";
  attribute NLW_MACRO_TAG : integer;
  attribute NLW_MACRO_TAG of HBx : label is 2;
  attribute NLW_UNIQUE_ID : integer;
  attribute NLW_UNIQUE_ID of HBx : label is 0;
begin
HBx: entity work.HandshakeBase
    port map (
      IClk => IClk,
      OClk => OClk,
      aReset(0) => aReset(0),
      iData(10 downto 0) => iData(10 downto 0),
      iPush(0) => iPush(0),
      iReady(0) => iReady(0),
      iStoredData(10 downto 0) => NLW_HBx_iStoredData_UNCONNECTED(10 downto 0),
      oData(10 downto 0) => oData(10 downto 0),
      oDataAck(0) => NLW_HBx_oDataAck_UNCONNECTED(0),
      oDataValid(0) => oDataValid(0)
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity Tdc is
  port (
    sEnableMeas : in STD_LOGIC;
    sMeasDoneRE : out STD_LOGIC;
    sTrigEnIn : in STD_LOGIC;
    sRolloverErr : out STD_LOGIC;
    FasterClk : in STD_LOGIC;
    SampleClk : in STD_LOGIC;
    aArmCounter : in STD_LOGIC;
    sCountValid : out STD_LOGIC;
    aReset : in STD_LOGIC;
    aTimeRefIn : in STD_LOGIC;
    sReadyToMeas : out STD_LOGIC;
    sCountOut : out STD_LOGIC_VECTOR ( 15 downto 0 )
  );
  attribute NotValidForBitStream : boolean;
  attribute NotValidForBitStream of Tdc : entity is true;
  attribute \TYPE\ : string;
  attribute \TYPE\ of Tdc : entity is "Tdc";
  attribute BUS_INFO : string;
  attribute BUS_INFO of Tdc : entity is "16:OUTPUT:sCountOut(15:0)";
  attribute KEEP_HIERARCHY : string;
  attribute KEEP_HIERARCHY of Tdc : entity is "TRUE";
  attribute NLW_UNIQUE_ID : integer;
  attribute NLW_UNIQUE_ID of Tdc : entity is 0;
  attribute NLW_MACRO_TAG : integer;
  attribute NLW_MACRO_TAG of Tdc : entity is 0;
  attribute NLW_MACRO_ALIAS : string;
  attribute NLW_MACRO_ALIAS of Tdc : entity is "Tdc_Tdc";
end Tdc;

architecture STRUCTURE of Tdc is
  signal FasterClk_IBUF : STD_LOGIC;
  signal Madd_sCountOutFlat_cy : STD_LOGIC_VECTOR ( 2 to 2 );
  signal Madd_sCountOutFlat_lut : STD_LOGIC_VECTOR ( 8 downto 3 );
  signal Madd_sCountOutMinus_cy : STD_LOGIC_VECTOR ( 1 to 1 );
  signal N02 : STD_LOGIC;
  signal N11 : STD_LOGIC;
  signal N13 : STD_LOGIC;
  signal N14 : STD_LOGIC;
  signal N181 : STD_LOGIC;
  signal N19 : STD_LOGIC;
  signal N5 : STD_LOGIC;
  signal N81 : STD_LOGIC;
  signal SampleClk_IBUF : STD_LOGIC;
  signal aArmCounter_IBUF : STD_LOGIC;
  attribute TIG : string;
  attribute TIG of aArmCounter_IBUF : signal is "";
  signal aReset_IBUF : STD_LOGIC;
  signal aTimeRefIn_IBUF : STD_LOGIC;
  signal \^farmcounter\ : STD_LOGIC;
  signal \^farmcounterdly\ : STD_LOGIC;
  signal \^farmcounter_ms\ : STD_LOGIC;
  signal fCount : STD_LOGIC_VECTOR ( 7 downto 0 );
  signal \^fcounthspush\ : STD_LOGIC;
  signal fCountHsPush_mux0000 : STD_LOGIC;
  signal fCountHsReadyBool : STD_LOGIC;
  signal \^fcountmsbset\ : STD_LOGIC;
  signal fCountMsbSet_mux0000 : STD_LOGIC;
  signal fCountOut : STD_LOGIC_VECTOR ( 7 downto 0 );
  signal fCountOut_mux0000 : STD_LOGIC_VECTOR ( 7 downto 0 );
  signal fCount_mux0000 : STD_LOGIC_VECTOR ( 7 downto 0 );
  signal \^fcount_mux0000(1)1\ : STD_LOGIC;
  signal \^fcount_mux0000(1)2\ : STD_LOGIC;
  signal \^fcount_mux0000(4)1\ : STD_LOGIC;
  signal \^fcount_mux0000(4)2\ : STD_LOGIC;
  signal \^fenablemeas\ : STD_LOGIC;
  signal \^fenablemeas_ms\ : STD_LOGIC;
  signal \^fextrahalf\ : STD_LOGIC;
  signal fExtraHalf_mux0000 : STD_LOGIC;
  signal \^fextrahalf_mux00001\ : STD_LOGIC;
  signal \^fextrahalf_mux00002\ : STD_LOGIC;
  signal fIddrQ1 : STD_LOGIC;
  signal \^fiddrq1dly\ : STD_LOGIC;
  signal fIddrQ2 : STD_LOGIC;
  signal \^fiddrq2dly\ : STD_LOGIC;
  signal \^fmeasdone\ : STD_LOGIC;
  signal fMeasDone_mux0000 : STD_LOGIC;
  signal \^fminushalf\ : STD_LOGIC;
  signal fMinusHalf_mux0000 : STD_LOGIC;
  signal \^freadytomeas\ : STD_LOGIC;
  signal fReadyToMeas_mux0001 : STD_LOGIC;
  signal \^frollovererr\ : STD_LOGIC;
  signal fRolloverErr_mux0000 : STD_LOGIC;
  signal \^ftdcstate_fsm_ffd1\ : STD_LOGIC;
  signal \fTdcState_FSM_FFd1-In\ : STD_LOGIC;
  signal \^ftdcstate_fsm_ffd1-in1\ : STD_LOGIC;
  signal \^ftdcstate_fsm_ffd1-in2\ : STD_LOGIC;
  signal \^ftdcstate_fsm_ffd2\ : STD_LOGIC;
  signal \fTdcState_FSM_FFd2-In\ : STD_LOGIC;
  signal \^ftrigeninfalledge\ : STD_LOGIC;
  signal \^ftrigeninfalledgedly\ : STD_LOGIC;
  signal \^ftrigenrisedge\ : STD_LOGIC;
  signal \^ftrigenrisedgedly\ : STD_LOGIC;
  signal \sCountOut[0]\ : STD_LOGIC;
  signal \sCountOut[10]\ : STD_LOGIC;
  signal \sCountOut[11]\ : STD_LOGIC;
  signal \sCountOut[12]\ : STD_LOGIC;
  signal \sCountOut[13]\ : STD_LOGIC;
  signal \sCountOut[14]\ : STD_LOGIC;
  signal \sCountOut[15]\ : STD_LOGIC;
  signal \sCountOut[1]\ : STD_LOGIC;
  signal \sCountOut[2]\ : STD_LOGIC;
  signal \sCountOut[3]\ : STD_LOGIC;
  signal \sCountOut[4]\ : STD_LOGIC;
  signal \sCountOut[5]\ : STD_LOGIC;
  signal \sCountOut[6]\ : STD_LOGIC;
  signal \sCountOut[7]\ : STD_LOGIC;
  signal \sCountOut[8]\ : STD_LOGIC;
  signal \sCountOut[9]\ : STD_LOGIC;
  signal sCountOut_OBUF : STD_LOGIC_VECTOR ( 15 downto 0 );
  signal sCountOut_mux0002 : STD_LOGIC_VECTOR ( 9 downto 1 );
  signal sCountOut_xor0000 : STD_LOGIC;
  signal \^scountvaliddly\ : STD_LOGIC;
  signal sCountValidInt : STD_LOGIC;
  signal sCountValid_OBUF : STD_LOGIC;
  signal sCountWord : STD_LOGIC_VECTOR ( 9 downto 8 );
  signal \^senablemeasreg\ : STD_LOGIC;
  signal sEnableMeas_IBUF : STD_LOGIC;
  signal \^smeasdone\ : STD_LOGIC;
  signal \^smeasdonedly\ : STD_LOGIC;
  signal sMeasDoneRE_OBUF : STD_LOGIC;
  signal sMeasDoneRE_and0000 : STD_LOGIC;
  signal sReadyToMeas_OBUF : STD_LOGIC;
  signal \^sreadytomeas_ms\ : STD_LOGIC;
  signal sRolloverErr_OBUF : STD_LOGIC;
  signal sTrigEnIn_IBUF : STD_LOGIC;
  signal \^strigenreg\ : STD_LOGIC;
  attribute OPT_INSERTED : boolean;
  attribute OPT_INSERTED of FasterClk_IBUF_inst : label is std.standard.true;
  attribute XSTLIB : boolean;
  attribute XSTLIB of Mxor_sCountOut_xor0000_Result1 : label is std.standard.true;
  attribute OPT_INSERTED of SampleClk_IBUF_inst : label is std.standard.true;
  attribute BUS_INFO of TdcHandshake : label is "1:INPUT:iPush(0:0)";
  attribute KEEP_HIERARCHY of TdcHandshake : label is "TRUE";
  attribute NLW_MACRO_ALIAS of TdcHandshake : label is "HandshakeSLV_TdcHandshake";
  attribute NLW_MACRO_TAG of TdcHandshake : label is 1;
  attribute NLW_UNIQUE_ID of TdcHandshake : label is 0;
  attribute XSTLIB of TimeRefSamplerIddr : label is std.standard.true;
  attribute \__SRVAL\ : string;
  attribute \__SRVAL\ of TimeRefSamplerIddr : label is "FALSE";
  attribute XSTLIB of XST_GND : label is std.standard.true;
  attribute OPT_INSERTED of aArmCounter_IBUF_inst : label is std.standard.true;
  attribute OPT_INSERTED of aReset_IBUF_inst : label is std.standard.true;
  attribute OPT_INSERTED of aTimeRefIn_IBUF_inst : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM : string;
  attribute XILINX_LEGACY_PRIM of fArmCounter : label is "FDC";
  attribute XSTLIB of fArmCounter : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fArmCounterDly : label is "FDC";
  attribute XSTLIB of fArmCounterDly : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fArmCounter_ms : label is "FDC";
  attribute XSTLIB of fArmCounter_ms : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fCountHsPush : label is "FDC";
  attribute XSTLIB of fCountHsPush : label is std.standard.true;
  attribute XSTLIB of fCountHsPush_mux00001 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fCountMsbSet : label is "FDC";
  attribute XSTLIB of fCountMsbSet : label is std.standard.true;
  attribute XSTLIB of fCountMsbSet_mux00001 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fCountOut_0 : label is "FDC";
  attribute XSTLIB of fCountOut_0 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fCountOut_1 : label is "FDC";
  attribute XSTLIB of fCountOut_1 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fCountOut_2 : label is "FDC";
  attribute XSTLIB of fCountOut_2 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fCountOut_3 : label is "FDC";
  attribute XSTLIB of fCountOut_3 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fCountOut_4 : label is "FDC";
  attribute XSTLIB of fCountOut_4 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fCountOut_5 : label is "FDC";
  attribute XSTLIB of fCountOut_5 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fCountOut_6 : label is "FDC";
  attribute XSTLIB of fCountOut_6 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fCountOut_7 : label is "FDC";
  attribute XSTLIB of fCountOut_7 : label is std.standard.true;
  attribute XSTLIB of \fCountOut_mux0000(0)1\ : label is std.standard.true;
  attribute XSTLIB of \fCountOut_mux0000(1)1\ : label is std.standard.true;
  attribute XSTLIB of \fCountOut_mux0000(2)1\ : label is std.standard.true;
  attribute XSTLIB of \fCountOut_mux0000(3)1\ : label is std.standard.true;
  attribute XSTLIB of \fCountOut_mux0000(4)1\ : label is std.standard.true;
  attribute XSTLIB of \fCountOut_mux0000(5)1\ : label is std.standard.true;
  attribute XSTLIB of \fCountOut_mux0000(6)1\ : label is std.standard.true;
  attribute XSTLIB of \fCountOut_mux0000(7)1\ : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fCount_0 : label is "FDC";
  attribute XSTLIB of fCount_0 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fCount_1 : label is "FDC";
  attribute XSTLIB of fCount_1 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fCount_2 : label is "FDC";
  attribute XSTLIB of fCount_2 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fCount_3 : label is "FDC";
  attribute XSTLIB of fCount_3 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fCount_4 : label is "FDC";
  attribute XSTLIB of fCount_4 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fCount_5 : label is "FDC";
  attribute XSTLIB of fCount_5 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fCount_6 : label is "FDC";
  attribute XSTLIB of fCount_6 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fCount_7 : label is "FDC";
  attribute XSTLIB of fCount_7 : label is std.standard.true;
  attribute XSTLIB of \fCount_mux0000(0)\ : label is std.standard.true;
  attribute XSTLIB of \fCount_mux0000(0)41\ : label is std.standard.true;
  attribute XSTLIB of \fCount_mux0000(0)_F\ : label is std.standard.true;
  attribute XSTLIB of \fCount_mux0000(0)_G\ : label is std.standard.true;
  attribute XSTLIB of \fCount_mux0000(1)1\ : label is std.standard.true;
  attribute XSTLIB of \fCount_mux0000(1)2\ : label is std.standard.true;
  attribute XSTLIB of \fCount_mux0000(1)_f7\ : label is std.standard.true;
  attribute XSTLIB of \fCount_mux0000(2)11\ : label is std.standard.true;
  attribute XSTLIB of \fCount_mux0000(2)2\ : label is std.standard.true;
  attribute XSTLIB of \fCount_mux0000(3)\ : label is std.standard.true;
  attribute XSTLIB of \fCount_mux0000(3)_SW0\ : label is std.standard.true;
  attribute XSTLIB of \fCount_mux0000(4)1\ : label is std.standard.true;
  attribute XSTLIB of \fCount_mux0000(4)2\ : label is std.standard.true;
  attribute XSTLIB of \fCount_mux0000(4)_f7\ : label is std.standard.true;
  attribute XSTLIB of \fCount_mux0000(5)1\ : label is std.standard.true;
  attribute XSTLIB of \fCount_mux0000(6)11\ : label is std.standard.true;
  attribute XSTLIB of \fCount_mux0000(6)2\ : label is std.standard.true;
  attribute XSTLIB of \fCount_mux0000(7)11\ : label is std.standard.true;
  attribute XSTLIB of \fCount_mux0000(7)2\ : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fEnableMeas : label is "FDC";
  attribute XSTLIB of fEnableMeas : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fEnableMeas_ms : label is "FDC";
  attribute XSTLIB of fEnableMeas_ms : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fExtraHalf : label is "FDC";
  attribute XSTLIB of fExtraHalf : label is std.standard.true;
  attribute XSTLIB of fExtraHalf_mux00001 : label is std.standard.true;
  attribute XSTLIB of fExtraHalf_mux00002 : label is std.standard.true;
  attribute XSTLIB of fExtraHalf_mux0000_f7 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fIddrQ1dly : label is "FDC";
  attribute XSTLIB of fIddrQ1dly : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fIddrQ2Dly : label is "FDC";
  attribute XSTLIB of fIddrQ2Dly : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fMeasDone : label is "FDC";
  attribute XSTLIB of fMeasDone : label is std.standard.true;
  attribute HU_SET : string;
  attribute HU_SET of fMeasDone : label is "sDoneUset";
  attribute RLOC : string;
  attribute RLOC of fMeasDone : label is "X0Y0";
  attribute XSTLIB of fMeasDone_mux00001 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fMinusHalf : label is "FDC";
  attribute XSTLIB of fMinusHalf : label is std.standard.true;
  attribute XSTLIB of fMinusHalf_mux00001 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fReadyToMeas : label is "FDC";
  attribute XSTLIB of fReadyToMeas : label is std.standard.true;
  attribute XSTLIB of fReadyToMeas_mux00011 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fRolloverErr : label is "FDC";
  attribute XSTLIB of fRolloverErr : label is std.standard.true;
  attribute XSTLIB of fRolloverErr_mux00001 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fTdcState_FSM_FFd1 : label is "FDC";
  attribute XSTLIB of fTdcState_FSM_FFd1 : label is std.standard.true;
  attribute XSTLIB of \fTdcState_FSM_FFd1-In1\ : label is std.standard.true;
  attribute XSTLIB of \fTdcState_FSM_FFd1-In2\ : label is std.standard.true;
  attribute XSTLIB of \fTdcState_FSM_FFd1-In_f7\ : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fTdcState_FSM_FFd2 : label is "FDC";
  attribute XSTLIB of fTdcState_FSM_FFd2 : label is std.standard.true;
  attribute XSTLIB of \fTdcState_FSM_FFd2-In1\ : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of fTrigEnInFallEdge : label is "FDC_1";
  attribute XSTLIB of fTrigEnInFallEdge : label is std.standard.true;
  attribute HU_SET of fTrigEnInFallEdge : label is "sTrigUset";
  attribute RLOC of fTrigEnInFallEdge : label is "X0Y2";
  attribute XILINX_LEGACY_PRIM of fTrigEnInFallEdgeDly : label is "FDC";
  attribute XSTLIB of fTrigEnInFallEdgeDly : label is std.standard.true;
  attribute HU_SET of fTrigEnInFallEdgeDly : label is "sTrigUset";
  attribute RLOC of fTrigEnInFallEdgeDly : label is "X0Y3";
  attribute XILINX_LEGACY_PRIM of fTrigEnRisEdge : label is "FDC";
  attribute XSTLIB of fTrigEnRisEdge : label is std.standard.true;
  attribute HU_SET of fTrigEnRisEdge : label is "sTrigUset";
  attribute RLOC of fTrigEnRisEdge : label is "X0Y0";
  attribute XILINX_LEGACY_PRIM of fTrigEnRisEdgeDly : label is "FDC";
  attribute XSTLIB of fTrigEnRisEdgeDly : label is std.standard.true;
  attribute OPT_INSERTED of \sCountOut[0]_OBUF_inst\ : label is std.standard.true;
  attribute OPT_INSERTED of \sCountOut[10]_OBUF_inst\ : label is std.standard.true;
  attribute OPT_INSERTED of \sCountOut[11]_OBUF_inst\ : label is std.standard.true;
  attribute OPT_INSERTED of \sCountOut[12]_OBUF_inst\ : label is std.standard.true;
  attribute OPT_INSERTED of \sCountOut[13]_OBUF_inst\ : label is std.standard.true;
  attribute OPT_INSERTED of \sCountOut[14]_OBUF_inst\ : label is std.standard.true;
  attribute OPT_INSERTED of \sCountOut[15]_OBUF_inst\ : label is std.standard.true;
  attribute OPT_INSERTED of \sCountOut[1]_OBUF_inst\ : label is std.standard.true;
  attribute OPT_INSERTED of \sCountOut[2]_OBUF_inst\ : label is std.standard.true;
  attribute OPT_INSERTED of \sCountOut[3]_OBUF_inst\ : label is std.standard.true;
  attribute OPT_INSERTED of \sCountOut[4]_OBUF_inst\ : label is std.standard.true;
  attribute OPT_INSERTED of \sCountOut[5]_OBUF_inst\ : label is std.standard.true;
  attribute OPT_INSERTED of \sCountOut[6]_OBUF_inst\ : label is std.standard.true;
  attribute OPT_INSERTED of \sCountOut[7]_OBUF_inst\ : label is std.standard.true;
  attribute OPT_INSERTED of \sCountOut[8]_OBUF_inst\ : label is std.standard.true;
  attribute OPT_INSERTED of \sCountOut[9]_OBUF_inst\ : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of sCountOut_0 : label is "FDC";
  attribute XSTLIB of sCountOut_0 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of sCountOut_1 : label is "FDC";
  attribute XSTLIB of sCountOut_1 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of sCountOut_2 : label is "FDC";
  attribute XSTLIB of sCountOut_2 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of sCountOut_3 : label is "FDC";
  attribute XSTLIB of sCountOut_3 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of sCountOut_4 : label is "FDC";
  attribute XSTLIB of sCountOut_4 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of sCountOut_5 : label is "FDC";
  attribute XSTLIB of sCountOut_5 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of sCountOut_6 : label is "FDC";
  attribute XSTLIB of sCountOut_6 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of sCountOut_7 : label is "FDC";
  attribute XSTLIB of sCountOut_7 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of sCountOut_8 : label is "FDC";
  attribute XSTLIB of sCountOut_8 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of sCountOut_9 : label is "FDC";
  attribute XSTLIB of sCountOut_9 : label is std.standard.true;
  attribute XSTLIB of \sCountOut_mux0002(1)1\ : label is std.standard.true;
  attribute XSTLIB of \sCountOut_mux0002(2)1\ : label is std.standard.true;
  attribute XSTLIB of \sCountOut_mux0002(3)1\ : label is std.standard.true;
  attribute XSTLIB of \sCountOut_mux0002(4)11\ : label is std.standard.true;
  attribute XSTLIB of \sCountOut_mux0002(4)2\ : label is std.standard.true;
  attribute XSTLIB of \sCountOut_mux0002(5)1\ : label is std.standard.true;
  attribute XSTLIB of \sCountOut_mux0002(6)1\ : label is std.standard.true;
  attribute XSTLIB of \sCountOut_mux0002(7)1\ : label is std.standard.true;
  attribute XSTLIB of \sCountOut_mux0002(8)1\ : label is std.standard.true;
  attribute XSTLIB of \sCountOut_mux0002(9)1\ : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of scountvalid_RnM : label is "FDC";
  attribute XSTLIB of scountvalid_RnM : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of sCountValidDly : label is "FDC";
  attribute XSTLIB of sCountValidDly : label is std.standard.true;
  attribute OPT_INSERTED of sCountValid_OBUF_inst : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of sEnableMeasReg : label is "FDC";
  attribute XSTLIB of sEnableMeasReg : label is std.standard.true;
  attribute OPT_INSERTED of sEnableMeas_IBUF_inst : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of sMeasDone : label is "FDC";
  attribute XSTLIB of sMeasDone : label is std.standard.true;
  attribute HU_SET of sMeasDone : label is "sDoneUset";
  attribute RLOC of sMeasDone : label is "X0Y1";
  attribute XILINX_LEGACY_PRIM of sMeasDoneDly : label is "FDC";
  attribute XSTLIB of sMeasDoneDly : label is std.standard.true;
  attribute HU_SET of sMeasDoneDly : label is "sDoneUset";
  attribute RLOC of sMeasDoneDly : label is "X0Y2";
  attribute XILINX_LEGACY_PRIM of smeasdonere_RnM : label is "FDC";
  attribute XSTLIB of smeasdonere_RnM : label is std.standard.true;
  attribute HU_SET of smeasdonere_RnM : label is "sDoneUset";
  attribute RLOC of smeasdonere_RnM : label is "X0Y2";
  attribute OPT_INSERTED of sMeasDoneRE_OBUF_inst : label is std.standard.true;
  attribute XSTLIB of sMeasDoneRE_and00001 : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of sreadytomeas_RnM : label is "FDC";
  attribute XSTLIB of sreadytomeas_RnM : label is std.standard.true;
  attribute OPT_INSERTED of sReadyToMeas_OBUF_inst : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of sReadyToMeas_ms : label is "FDC";
  attribute XSTLIB of sReadyToMeas_ms : label is std.standard.true;
  attribute OPT_INSERTED of sRolloverErr_OBUF_inst : label is std.standard.true;
  attribute OPT_INSERTED of sTrigEnIn_IBUF_inst : label is std.standard.true;
  attribute XILINX_LEGACY_PRIM of sTrigEnReg : label is "FDC";
  attribute XSTLIB of sTrigEnReg : label is std.standard.true;
  attribute HU_SET of sTrigEnReg : label is "sTrigUset";
  attribute RLOC of sTrigEnReg : label is "X0Y1";
begin
  sCountOut(15) <= \sCountOut[15]\;
  sCountOut(14) <= \sCountOut[14]\;
  sCountOut(13) <= \sCountOut[13]\;
  sCountOut(12) <= \sCountOut[12]\;
  sCountOut(11) <= \sCountOut[11]\;
  sCountOut(10) <= \sCountOut[10]\;
  sCountOut(9) <= \sCountOut[9]\;
  sCountOut(8) <= \sCountOut[8]\;
  sCountOut(7) <= \sCountOut[7]\;
  sCountOut(6) <= \sCountOut[6]\;
  sCountOut(5) <= \sCountOut[5]\;
  sCountOut(4) <= \sCountOut[4]\;
  sCountOut(3) <= \sCountOut[3]\;
  sCountOut(2) <= \sCountOut[2]\;
  sCountOut(1) <= \sCountOut[1]\;
  sCountOut(0) <= \sCountOut[0]\;
FasterClk_IBUF_inst: unisim.vcomponents.IBUF
    port map (
      I => FasterClk,
      O => FasterClk_IBUF
    );
Mxor_sCountOut_xor0000_Result1: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
    port map (
      I0 => sCountWord(9),
      I1 => sCountWord(8),
      O => sCountOut_xor0000
    );
SampleClk_IBUF_inst: unisim.vcomponents.IBUF
    port map (
      I => SampleClk,
      O => SampleClk_IBUF
    );
TdcHandshake: entity work.HandshakeSLV
    port map (
      IClk => FasterClk_IBUF,
      OClk => SampleClk_IBUF,
      aReset(0) => aReset_IBUF,
      iData(10) => \^frollovererr\,
      iData(9) => \^fextrahalf\,
      iData(8) => \^fminushalf\,
      iData(7 downto 0) => fCountOut(7 downto 0),
      iPush(0) => \^fcounthspush\,
      iReady(0) => fCountHsReadyBool,
      oData(10) => sRolloverErr_OBUF,
      oData(9 downto 8) => sCountWord(9 downto 8),
      oData(7 downto 2) => Madd_sCountOutFlat_lut(8 downto 3),
      oData(1) => Madd_sCountOutFlat_cy(2),
      oData(0) => Madd_sCountOutMinus_cy(1),
      oDataValid(0) => sCountValidInt
    );
TimeRefSamplerIddr: unisim.vcomponents.IDDR_2CLK
    generic map(
      DDR_CLK_EDGE => "SAME_EDGE_PIPELINED",
      INIT_Q1 => '0',
      INIT_Q2 => '0',
      IS_CB_INVERTED => '1',
      SRTYPE => "ASYNC"
    )
    port map (
      C => FasterClk_IBUF,
      CB => FasterClk_IBUF,
      CE => '1',
      D => aTimeRefIn_IBUF,
      Q1 => fIddrQ1,
      Q2 => fIddrQ2,
      R => aReset_IBUF,
      S => sCountOut_OBUF(10)
    );
XST_GND: unisim.vcomponents.GND
    port map (
      G => sCountOut_OBUF(10)
    );
aArmCounter_IBUF_inst: unisim.vcomponents.IBUF
    port map (
      I => aArmCounter,
      O => aArmCounter_IBUF
    );
aReset_IBUF_inst: unisim.vcomponents.IBUF
    port map (
      I => aReset,
      O => aReset_IBUF
    );
aTimeRefIn_IBUF_inst: unisim.vcomponents.IBUF
    port map (
      I => aTimeRefIn,
      O => aTimeRefIn_IBUF
    );
fArmCounter: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => \^farmcounter_ms\,
      Q => \^farmcounter\
    );
fArmCounterDly: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => \^farmcounter\,
      Q => \^farmcounterdly\
    );
fArmCounter_ms: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => aArmCounter_IBUF,
      Q => \^farmcounter_ms\
    );
fCountHsPush: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fCountHsPush_mux0000,
      Q => \^fcounthspush\
    );
fCountHsPush_mux00001: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AAAAAAAA00008000"
    )
    port map (
      I0 => \^ftdcstate_fsm_ffd2\,
      I1 => \^ftdcstate_fsm_ffd1\,
      I2 => \^fenablemeas\,
      I3 => \^ftrigenrisedge\,
      I4 => \^ftrigenrisedgedly\,
      I5 => \^fcounthspush\,
      O => fCountHsPush_mux0000
    );
fCountMsbSet: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fCountMsbSet_mux0000,
      Q => \^fcountmsbset\
    );
fCountMsbSet_mux00001: unisim.vcomponents.LUT4
    generic map(
      INIT => X"E8C8"
    )
    port map (
      I0 => \^ftdcstate_fsm_ffd1\,
      I1 => \^fcountmsbset\,
      I2 => \^ftdcstate_fsm_ffd2\,
      I3 => fCount(7),
      O => fCountMsbSet_mux0000
    );
fCountOut_0: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fCountOut_mux0000(7),
      Q => fCountOut(0)
    );
fCountOut_1: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fCountOut_mux0000(6),
      Q => fCountOut(1)
    );
fCountOut_2: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fCountOut_mux0000(5),
      Q => fCountOut(2)
    );
fCountOut_3: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fCountOut_mux0000(4),
      Q => fCountOut(3)
    );
fCountOut_4: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fCountOut_mux0000(3),
      Q => fCountOut(4)
    );
fCountOut_5: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fCountOut_mux0000(2),
      Q => fCountOut(5)
    );
fCountOut_6: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fCountOut_mux0000(1),
      Q => fCountOut(6)
    );
fCountOut_7: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fCountOut_mux0000(0),
      Q => fCountOut(7)
    );
\fCountOut_mux0000(0)1\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"A8E8A8A8A828A8A8"
    )
    port map (
      I0 => fCountOut(7),
      I1 => \^ftdcstate_fsm_ffd2\,
      I2 => \^ftdcstate_fsm_ffd1\,
      I3 => N02,
      I4 => \^fenablemeas\,
      I5 => fCount(7),
      O => fCountOut_mux0000(0)
    );
\fCountOut_mux0000(1)1\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"A8E8A8A8A828A8A8"
    )
    port map (
      I0 => fCountOut(6),
      I1 => \^ftdcstate_fsm_ffd2\,
      I2 => \^ftdcstate_fsm_ffd1\,
      I3 => N02,
      I4 => \^fenablemeas\,
      I5 => fCount(6),
      O => fCountOut_mux0000(1)
    );
\fCountOut_mux0000(2)1\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"A8E8A8A8A828A8A8"
    )
    port map (
      I0 => fCountOut(5),
      I1 => \^ftdcstate_fsm_ffd2\,
      I2 => \^ftdcstate_fsm_ffd1\,
      I3 => N02,
      I4 => \^fenablemeas\,
      I5 => fCount(5),
      O => fCountOut_mux0000(2)
    );
\fCountOut_mux0000(3)1\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"A8E8A8A8A828A8A8"
    )
    port map (
      I0 => fCountOut(4),
      I1 => \^ftdcstate_fsm_ffd2\,
      I2 => \^ftdcstate_fsm_ffd1\,
      I3 => N02,
      I4 => \^fenablemeas\,
      I5 => fCount(4),
      O => fCountOut_mux0000(3)
    );
\fCountOut_mux0000(4)1\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"A8E8A8A8A828A8A8"
    )
    port map (
      I0 => fCountOut(3),
      I1 => \^ftdcstate_fsm_ffd2\,
      I2 => \^ftdcstate_fsm_ffd1\,
      I3 => N02,
      I4 => \^fenablemeas\,
      I5 => fCount(3),
      O => fCountOut_mux0000(4)
    );
\fCountOut_mux0000(5)1\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"A8E8A8A8A828A8A8"
    )
    port map (
      I0 => fCountOut(2),
      I1 => \^ftdcstate_fsm_ffd2\,
      I2 => \^ftdcstate_fsm_ffd1\,
      I3 => N02,
      I4 => \^fenablemeas\,
      I5 => fCount(2),
      O => fCountOut_mux0000(5)
    );
\fCountOut_mux0000(6)1\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"A8E8A8A8A828A8A8"
    )
    port map (
      I0 => fCountOut(1),
      I1 => \^ftdcstate_fsm_ffd2\,
      I2 => \^ftdcstate_fsm_ffd1\,
      I3 => N02,
      I4 => \^fenablemeas\,
      I5 => fCount(1),
      O => fCountOut_mux0000(6)
    );
\fCountOut_mux0000(7)1\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"A8E8A8A8A828A8A8"
    )
    port map (
      I0 => fCountOut(0),
      I1 => \^ftdcstate_fsm_ffd2\,
      I2 => \^ftdcstate_fsm_ffd1\,
      I3 => N02,
      I4 => \^fenablemeas\,
      I5 => fCount(0),
      O => fCountOut_mux0000(7)
    );
fCount_0: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fCount_mux0000(7),
      Q => fCount(0)
    );
fCount_1: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fCount_mux0000(6),
      Q => fCount(1)
    );
fCount_2: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fCount_mux0000(5),
      Q => fCount(2)
    );
fCount_3: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fCount_mux0000(4),
      Q => fCount(3)
    );
fCount_4: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fCount_mux0000(3),
      Q => fCount(4)
    );
fCount_5: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fCount_mux0000(2),
      Q => fCount(5)
    );
fCount_6: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fCount_mux0000(1),
      Q => fCount(6)
    );
fCount_7: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fCount_mux0000(0),
      Q => fCount(7)
    );
\fCount_mux0000(0)\: unisim.vcomponents.MUXF7
    port map (
      I0 => N13,
      I1 => N14,
      O => fCount_mux0000(0),
      S => fCount(7)
    );
\fCount_mux0000(0)41\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0000800000000000"
    )
    port map (
      I0 => fCount(1),
      I1 => fCount(3),
      I2 => \^ftdcstate_fsm_ffd1\,
      I3 => fCount(0),
      I4 => N181,
      I5 => fCount(2),
      O => N19
    );
\fCount_mux0000(0)_F\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"8000"
    )
    port map (
      I0 => fCount(4),
      I1 => fCount(6),
      I2 => fCount(5),
      I3 => N19,
      O => N13
    );
\fCount_mux0000(0)_G\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"F7A2FFAA"
    )
    port map (
      I0 => \^ftdcstate_fsm_ffd1\,
      I1 => fCount(6),
      I2 => N5,
      I3 => \^ftdcstate_fsm_ffd2\,
      I4 => fCount(5),
      O => N14
    );
\fCount_mux0000(1)1\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"EAAAEAAAEAAA4000"
    )
    port map (
      I0 => fCount(6),
      I1 => fCount(5),
      I2 => fCount(4),
      I3 => N19,
      I4 => \^ftdcstate_fsm_ffd1\,
      I5 => \^ftdcstate_fsm_ffd2\,
      O => \^fcount_mux0000(1)1\
    );
\fCount_mux0000(1)2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"6222EAAA62224000"
    )
    port map (
      I0 => fCount(6),
      I1 => fCount(5),
      I2 => N19,
      I3 => fCount(4),
      I4 => \^ftdcstate_fsm_ffd1\,
      I5 => \^ftdcstate_fsm_ffd2\,
      O => \^fcount_mux0000(1)2\
    );
\fCount_mux0000(1)_f7\: unisim.vcomponents.MUXF7
    port map (
      I0 => \^fcount_mux0000(1)2\,
      I1 => \^fcount_mux0000(1)1\,
      O => fCount_mux0000(1),
      S => N5
    );
\fCount_mux0000(2)11\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"DFFFFFFFFFFFFFFF"
    )
    port map (
      I0 => fCount(4),
      I1 => N181,
      I2 => fCount(2),
      I3 => fCount(0),
      I4 => fCount(1),
      I5 => fCount(3),
      O => N5
    );
\fCount_mux0000(2)2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"F7D5A280A280A280"
    )
    port map (
      I0 => fCount(5),
      I1 => \^ftdcstate_fsm_ffd1\,
      I2 => N5,
      I3 => \^ftdcstate_fsm_ffd2\,
      I4 => N19,
      I5 => fCount(4),
      O => fCount_mux0000(2)
    );
\fCount_mux0000(3)\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FB3BC808FBFBC8C8"
    )
    port map (
      I0 => \^ftdcstate_fsm_ffd2\,
      I1 => fCount(4),
      I2 => \^ftdcstate_fsm_ffd1\,
      I3 => N181,
      I4 => N19,
      I5 => N11,
      O => fCount_mux0000(3)
    );
\fCount_mux0000(3)_SW0\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"8000"
    )
    port map (
      I0 => fCount(3),
      I1 => fCount(2),
      I2 => fCount(1),
      I3 => fCount(0),
      O => N11
    );
\fCount_mux0000(4)1\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"A8"
    )
    port map (
      I0 => fCount(3),
      I1 => \^ftdcstate_fsm_ffd1\,
      I2 => \^ftdcstate_fsm_ffd2\,
      O => \^fcount_mux0000(4)1\
    );
\fCount_mux0000(4)2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"6CCCCCCC28888888"
    )
    port map (
      I0 => \^ftdcstate_fsm_ffd1\,
      I1 => fCount(3),
      I2 => fCount(1),
      I3 => fCount(0),
      I4 => fCount(2),
      I5 => \^ftdcstate_fsm_ffd2\,
      O => \^fcount_mux0000(4)2\
    );
\fCount_mux0000(4)_f7\: unisim.vcomponents.MUXF7
    port map (
      I0 => \^fcount_mux0000(4)2\,
      I1 => \^fcount_mux0000(4)1\,
      O => fCount_mux0000(4),
      S => N181
    );
\fCount_mux0000(5)1\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"C6CCCCCC82888888"
    )
    port map (
      I0 => \^ftdcstate_fsm_ffd1\,
      I1 => fCount(2),
      I2 => N181,
      I3 => fCount(1),
      I4 => fCount(0),
      I5 => \^ftdcstate_fsm_ffd2\,
      O => fCount_mux0000(5)
    );
\fCount_mux0000(6)11\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"D"
    )
    port map (
      I0 => \^ftrigenrisedge\,
      I1 => \^ftrigenrisedgedly\,
      O => N02
    );
\fCount_mux0000(6)2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"68C8C8C8C8C8C8C8"
    )
    port map (
      I0 => \^ftdcstate_fsm_ffd2\,
      I1 => fCount(1),
      I2 => \^ftdcstate_fsm_ffd1\,
      I3 => \^fenablemeas\,
      I4 => N02,
      I5 => fCount(0),
      O => fCount_mux0000(6)
    );
\fCount_mux0000(7)11\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"77F7"
    )
    port map (
      I0 => \^fenablemeas\,
      I1 => \^ftdcstate_fsm_ffd2\,
      I2 => \^ftrigenrisedge\,
      I3 => \^ftrigenrisedgedly\,
      O => N181
    );
\fCount_mux0000(7)2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"6868C868C8C8C8C8"
    )
    port map (
      I0 => \^ftdcstate_fsm_ffd2\,
      I1 => fCount(0),
      I2 => \^ftdcstate_fsm_ffd1\,
      I3 => \^ftrigenrisedge\,
      I4 => \^ftrigenrisedgedly\,
      I5 => \^fenablemeas\,
      O => fCount_mux0000(7)
    );
fEnableMeas: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => \^fenablemeas_ms\,
      Q => \^fenablemeas\
    );
fEnableMeas_ms: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => \^senablemeasreg\,
      Q => \^fenablemeas_ms\
    );
fExtraHalf: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fExtraHalf_mux0000,
      Q => \^fextrahalf\
    );
fExtraHalf_mux00001: unisim.vcomponents.LUT6
    generic map(
      INIT => X"FFFBFFFF04000000"
    )
    port map (
      I0 => \^ftdcstate_fsm_ffd1\,
      I1 => \^fenablemeas\,
      I2 => \^fiddrq1dly\,
      I3 => \^fiddrq2dly\,
      I4 => fIddrQ1,
      I5 => \^fextrahalf\,
      O => \^fextrahalf_mux00001\
    );
fExtraHalf_mux00002: unisim.vcomponents.LUT2
    generic map(
      INIT => X"8"
    )
    port map (
      I0 => \^fextrahalf\,
      I1 => \^ftdcstate_fsm_ffd1\,
      O => \^fextrahalf_mux00002\
    );
fExtraHalf_mux0000_f7: unisim.vcomponents.MUXF7
    port map (
      I0 => \^fextrahalf_mux00002\,
      I1 => \^fextrahalf_mux00001\,
      O => fExtraHalf_mux0000,
      S => \^ftdcstate_fsm_ffd2\
    );
fIddrQ1dly: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fIddrQ1,
      Q => \^fiddrq1dly\
    );
fIddrQ2Dly: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fIddrQ2,
      Q => \^fiddrq2dly\
    );
fMeasDone: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fMeasDone_mux0000,
      Q => \^fmeasdone\
    );
fMeasDone_mux00001: unisim.vcomponents.LUT6
    generic map(
      INIT => X"A8A8E8A8A8A8A8A8"
    )
    port map (
      I0 => \^fmeasdone\,
      I1 => \^ftdcstate_fsm_ffd1\,
      I2 => \^ftdcstate_fsm_ffd2\,
      I3 => \^ftrigenrisedge\,
      I4 => \^ftrigenrisedgedly\,
      I5 => \^fenablemeas\,
      O => fMeasDone_mux0000
    );
fMinusHalf: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fMinusHalf_mux0000,
      Q => \^fminushalf\
    );
fMinusHalf_mux00001: unisim.vcomponents.LUT6
    generic map(
      INIT => X"A8E8A8A8A828A8A8"
    )
    port map (
      I0 => \^fminushalf\,
      I1 => \^ftdcstate_fsm_ffd2\,
      I2 => \^ftdcstate_fsm_ffd1\,
      I3 => N02,
      I4 => \^fenablemeas\,
      I5 => \^ftrigeninfalledgedly\,
      O => fMinusHalf_mux0000
    );
fReadyToMeas: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fReadyToMeas_mux0001,
      Q => \^freadytomeas\
    );
fReadyToMeas_mux00011: unisim.vcomponents.LUT3
    generic map(
      INIT => X"02"
    )
    port map (
      I0 => \^fenablemeas\,
      I1 => \^ftdcstate_fsm_ffd2\,
      I2 => \^ftdcstate_fsm_ffd1\,
      O => fReadyToMeas_mux0001
    );
fRolloverErr: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => fRolloverErr_mux0000,
      Q => \^frollovererr\
    );
fRolloverErr_mux00001: unisim.vcomponents.LUT5
    generic map(
      INIT => X"C8C8E8C8"
    )
    port map (
      I0 => \^ftdcstate_fsm_ffd2\,
      I1 => \^frollovererr\,
      I2 => \^ftdcstate_fsm_ffd1\,
      I3 => \^fcountmsbset\,
      I4 => fCount(7),
      O => fRolloverErr_mux0000
    );
fTdcState_FSM_FFd1: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => \fTdcState_FSM_FFd1-In\,
      Q => \^ftdcstate_fsm_ffd1\
    );
\fTdcState_FSM_FFd1-In1\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"8A88"
    )
    port map (
      I0 => \^fenablemeas\,
      I1 => \^ftdcstate_fsm_ffd1\,
      I2 => \^fiddrq1dly\,
      I3 => fIddrQ1,
      O => \^ftdcstate_fsm_ffd1-in1\
    );
\fTdcState_FSM_FFd1-In2\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"AAA2"
    )
    port map (
      I0 => \^ftdcstate_fsm_ffd1\,
      I1 => fCountHsReadyBool,
      I2 => \^farmcounter\,
      I3 => \^fcounthspush\,
      O => \^ftdcstate_fsm_ffd1-in2\
    );
\fTdcState_FSM_FFd1-In_f7\: unisim.vcomponents.MUXF7
    port map (
      I0 => \^ftdcstate_fsm_ffd1-in2\,
      I1 => \^ftdcstate_fsm_ffd1-in1\,
      O => \fTdcState_FSM_FFd1-In\,
      S => \^ftdcstate_fsm_ffd2\
    );
fTdcState_FSM_FFd2: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => \fTdcState_FSM_FFd2-In\,
      Q => \^ftdcstate_fsm_ffd2\
    );
\fTdcState_FSM_FFd2-In1\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"88888A8808080A08"
    )
    port map (
      I0 => \^fenablemeas\,
      I1 => \^ftdcstate_fsm_ffd2\,
      I2 => \^ftdcstate_fsm_ffd1\,
      I3 => \^farmcounter\,
      I4 => \^farmcounterdly\,
      I5 => N02,
      O => \fTdcState_FSM_FFd2-In\
    );
fTrigEnInFallEdge: unisim.vcomponents.FDCE
    generic map(
      INIT => '0',
      IS_C_INVERTED => '1'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => \^strigenreg\,
      Q => \^ftrigeninfalledge\
    );
fTrigEnInFallEdgeDly: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => \^ftrigeninfalledge\,
      Q => \^ftrigeninfalledgedly\
    );
fTrigEnRisEdge: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => \^strigenreg\,
      Q => \^ftrigenrisedge\
    );
fTrigEnRisEdgeDly: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => FasterClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => \^ftrigenrisedge\,
      Q => \^ftrigenrisedgedly\
    );
\sCountOut[0]_OBUF_inst\: unisim.vcomponents.OBUF
    port map (
      I => sCountOut_OBUF(0),
      O => \sCountOut[0]\
    );
\sCountOut[10]_OBUF_inst\: unisim.vcomponents.OBUF
    port map (
      I => sCountOut_OBUF(10),
      O => \sCountOut[10]\
    );
\sCountOut[11]_OBUF_inst\: unisim.vcomponents.OBUF
    port map (
      I => sCountOut_OBUF(10),
      O => \sCountOut[11]\
    );
\sCountOut[12]_OBUF_inst\: unisim.vcomponents.OBUF
    port map (
      I => sCountOut_OBUF(10),
      O => \sCountOut[12]\
    );
\sCountOut[13]_OBUF_inst\: unisim.vcomponents.OBUF
    port map (
      I => sCountOut_OBUF(10),
      O => \sCountOut[13]\
    );
\sCountOut[14]_OBUF_inst\: unisim.vcomponents.OBUF
    port map (
      I => sCountOut_OBUF(10),
      O => \sCountOut[14]\
    );
\sCountOut[15]_OBUF_inst\: unisim.vcomponents.OBUF
    port map (
      I => sCountOut_OBUF(10),
      O => \sCountOut[15]\
    );
\sCountOut[1]_OBUF_inst\: unisim.vcomponents.OBUF
    port map (
      I => sCountOut_OBUF(1),
      O => \sCountOut[1]\
    );
\sCountOut[2]_OBUF_inst\: unisim.vcomponents.OBUF
    port map (
      I => sCountOut_OBUF(2),
      O => \sCountOut[2]\
    );
\sCountOut[3]_OBUF_inst\: unisim.vcomponents.OBUF
    port map (
      I => sCountOut_OBUF(3),
      O => \sCountOut[3]\
    );
\sCountOut[4]_OBUF_inst\: unisim.vcomponents.OBUF
    port map (
      I => sCountOut_OBUF(4),
      O => \sCountOut[4]\
    );
\sCountOut[5]_OBUF_inst\: unisim.vcomponents.OBUF
    port map (
      I => sCountOut_OBUF(5),
      O => \sCountOut[5]\
    );
\sCountOut[6]_OBUF_inst\: unisim.vcomponents.OBUF
    port map (
      I => sCountOut_OBUF(6),
      O => \sCountOut[6]\
    );
\sCountOut[7]_OBUF_inst\: unisim.vcomponents.OBUF
    port map (
      I => sCountOut_OBUF(7),
      O => \sCountOut[7]\
    );
\sCountOut[8]_OBUF_inst\: unisim.vcomponents.OBUF
    port map (
      I => sCountOut_OBUF(8),
      O => \sCountOut[8]\
    );
\sCountOut[9]_OBUF_inst\: unisim.vcomponents.OBUF
    port map (
      I => sCountOut_OBUF(9),
      O => \sCountOut[9]\
    );
sCountOut_0: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => SampleClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => sCountOut_xor0000,
      Q => sCountOut_OBUF(0)
    );
sCountOut_1: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => SampleClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => sCountOut_mux0002(1),
      Q => sCountOut_OBUF(1)
    );
sCountOut_2: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => SampleClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => sCountOut_mux0002(2),
      Q => sCountOut_OBUF(2)
    );
sCountOut_3: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => SampleClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => sCountOut_mux0002(3),
      Q => sCountOut_OBUF(3)
    );
sCountOut_4: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => SampleClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => sCountOut_mux0002(4),
      Q => sCountOut_OBUF(4)
    );
sCountOut_5: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => SampleClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => sCountOut_mux0002(5),
      Q => sCountOut_OBUF(5)
    );
sCountOut_6: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => SampleClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => sCountOut_mux0002(6),
      Q => sCountOut_OBUF(6)
    );
sCountOut_7: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => SampleClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => sCountOut_mux0002(7),
      Q => sCountOut_OBUF(7)
    );
sCountOut_8: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => SampleClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => sCountOut_mux0002(8),
      Q => sCountOut_OBUF(8)
    );
sCountOut_9: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => SampleClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => sCountOut_mux0002(9),
      Q => sCountOut_OBUF(9)
    );
\sCountOut_mux0002(1)1\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"9A"
    )
    port map (
      I0 => Madd_sCountOutMinus_cy(1),
      I1 => sCountWord(9),
      I2 => sCountWord(8),
      O => sCountOut_mux0002(1)
    );
\sCountOut_mux0002(2)1\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"5655"
    )
    port map (
      I0 => Madd_sCountOutFlat_cy(2),
      I1 => Madd_sCountOutMinus_cy(1),
      I2 => sCountWord(9),
      I3 => sCountWord(8),
      O => sCountOut_mux0002(2)
    );
\sCountOut_mux0002(3)1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"66666C66"
    )
    port map (
      I0 => Madd_sCountOutFlat_cy(2),
      I1 => Madd_sCountOutFlat_lut(3),
      I2 => Madd_sCountOutMinus_cy(1),
      I3 => sCountWord(8),
      I4 => sCountWord(9),
      O => sCountOut_mux0002(3)
    );
\sCountOut_mux0002(4)11\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"02FFFFFF"
    )
    port map (
      I0 => sCountWord(8),
      I1 => sCountWord(9),
      I2 => Madd_sCountOutMinus_cy(1),
      I3 => Madd_sCountOutFlat_lut(3),
      I4 => Madd_sCountOutFlat_cy(2),
      O => N81
    );
\sCountOut_mux0002(4)2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"5AAA6AAA5AAA5AAA"
    )
    port map (
      I0 => Madd_sCountOutFlat_lut(4),
      I1 => Madd_sCountOutMinus_cy(1),
      I2 => Madd_sCountOutFlat_cy(2),
      I3 => Madd_sCountOutFlat_lut(3),
      I4 => sCountWord(9),
      I5 => sCountWord(8),
      O => sCountOut_mux0002(4)
    );
\sCountOut_mux0002(5)1\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"9A"
    )
    port map (
      I0 => Madd_sCountOutFlat_lut(5),
      I1 => N81,
      I2 => Madd_sCountOutFlat_lut(4),
      O => sCountOut_mux0002(5)
    );
\sCountOut_mux0002(6)1\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"9AAA"
    )
    port map (
      I0 => Madd_sCountOutFlat_lut(6),
      I1 => N81,
      I2 => Madd_sCountOutFlat_lut(5),
      I3 => Madd_sCountOutFlat_lut(4),
      O => sCountOut_mux0002(6)
    );
\sCountOut_mux0002(7)1\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"9AAAAAAA"
    )
    port map (
      I0 => Madd_sCountOutFlat_lut(7),
      I1 => N81,
      I2 => Madd_sCountOutFlat_lut(6),
      I3 => Madd_sCountOutFlat_lut(5),
      I4 => Madd_sCountOutFlat_lut(4),
      O => sCountOut_mux0002(7)
    );
\sCountOut_mux0002(8)1\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"9AAAAAAAAAAAAAAA"
    )
    port map (
      I0 => Madd_sCountOutFlat_lut(8),
      I1 => N81,
      I2 => Madd_sCountOutFlat_lut(7),
      I3 => Madd_sCountOutFlat_lut(6),
      I4 => Madd_sCountOutFlat_lut(5),
      I5 => Madd_sCountOutFlat_lut(4),
      O => sCountOut_mux0002(8)
    );
\sCountOut_mux0002(9)1\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0000800000000000"
    )
    port map (
      I0 => Madd_sCountOutFlat_lut(7),
      I1 => Madd_sCountOutFlat_lut(8),
      I2 => Madd_sCountOutFlat_lut(5),
      I3 => Madd_sCountOutFlat_lut(6),
      I4 => N81,
      I5 => Madd_sCountOutFlat_lut(4),
      O => sCountOut_mux0002(9)
    );
scountvalid_RnM: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => SampleClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => \^scountvaliddly\,
      Q => sCountValid_OBUF
    );
sCountValidDly: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => SampleClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => sCountValidInt,
      Q => \^scountvaliddly\
    );
sCountValid_OBUF_inst: unisim.vcomponents.OBUF
    port map (
      I => sCountValid_OBUF,
      O => sCountValid
    );
sEnableMeasReg: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => SampleClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => sEnableMeas_IBUF,
      Q => \^senablemeasreg\
    );
sEnableMeas_IBUF_inst: unisim.vcomponents.IBUF
    port map (
      I => sEnableMeas,
      O => sEnableMeas_IBUF
    );
sMeasDone: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => SampleClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => \^fmeasdone\,
      Q => \^smeasdone\
    );
sMeasDoneDly: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => SampleClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => \^smeasdone\,
      Q => \^smeasdonedly\
    );
smeasdonere_RnM: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => SampleClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => sMeasDoneRE_and0000,
      Q => sMeasDoneRE_OBUF
    );
sMeasDoneRE_OBUF_inst: unisim.vcomponents.OBUF
    port map (
      I => sMeasDoneRE_OBUF,
      O => sMeasDoneRE
    );
sMeasDoneRE_and00001: unisim.vcomponents.LUT2
    generic map(
      INIT => X"2"
    )
    port map (
      I0 => \^smeasdone\,
      I1 => \^smeasdonedly\,
      O => sMeasDoneRE_and0000
    );
sreadytomeas_RnM: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => SampleClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => \^sreadytomeas_ms\,
      Q => sReadyToMeas_OBUF
    );
sReadyToMeas_OBUF_inst: unisim.vcomponents.OBUF
    port map (
      I => sReadyToMeas_OBUF,
      O => sReadyToMeas
    );
sReadyToMeas_ms: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => SampleClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => \^freadytomeas\,
      Q => \^sreadytomeas_ms\
    );
sRolloverErr_OBUF_inst: unisim.vcomponents.OBUF
    port map (
      I => sRolloverErr_OBUF,
      O => sRolloverErr
    );
sTrigEnIn_IBUF_inst: unisim.vcomponents.IBUF
    port map (
      I => sTrigEnIn,
      O => sTrigEnIn_IBUF
    );
sTrigEnReg: unisim.vcomponents.FDCE
    generic map(
      INIT => '0'
    )
    port map (
      C => SampleClk_IBUF,
      CE => '1',
      CLR => aReset_IBUF,
      D => sTrigEnIn_IBUF,
      Q => \^strigenreg\
    );
end STRUCTURE;
