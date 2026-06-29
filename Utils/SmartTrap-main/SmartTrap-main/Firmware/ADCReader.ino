/*
Arduino sketch for reading the AD7616 analog digital converted using an arduino portenta.
Uses the parallel interface of the ADC.
*/
#define SEQEN       PI_10
#define RESET       PI_13
#define WR_BURST    PH_10
#define SCLK_RD     PH_12
#define CS          PI_4


#define CHSEL0      PI_7
#define CHSEL1      PI_5
#define CHSEL2      PC_13

#define HW_RNGSEL0  PA_4

#define AD_BUSY     PI_6
#define CONVST      PH_6

#define DB0         PB_15
#define DB1         PH_9
#define DB2         PB_3
#define DB3         PH_11
#define DB4         PB_4
#define DB5         PH_14
#define DB6         PB_14
#define DB7         PJ_8
#define DB8         PD_7
#define DB9         PA_11
#define DB10        PD_6
#define DB11        PG_14
#define DB12        PD_3
#define DB13        PG_9
#define DB14        PB_9
#define DB15        PI_2


#define READ_PIN(port, bit, outbit) (((port >> bit) & 1) << outbit)

void setup_ADC() {
  /*
  Initiates the analog digital convertedr by setting the various pins to their correct values
  and modes.
  */
  pinMode(LED_BUILTIN, OUTPUT);

  // Initalize the pins to output/input
  pinMode(SEQEN, OUTPUT);
  pinMode(RESET, OUTPUT);
  pinMode(WR_BURST, OUTPUT);
  pinMode(SCLK_RD, OUTPUT);
  pinMode(CS, OUTPUT);

  pinMode(HW_RNGSEL0, OUTPUT);

  pinMode(CHSEL0, OUTPUT);
  pinMode(CHSEL1, OUTPUT);
  pinMode(CHSEL2, OUTPUT);
  pinMode(AD_BUSY, INPUT);
  pinMode(CONVST, OUTPUT);

  // Set the paralell input pins
  pinMode(DB0,INPUT_PULLUP);  // Used to have just input on all these
  pinMode(DB1,INPUT_PULLUP);
  pinMode(DB2,INPUT_PULLUP);
  pinMode(DB3,INPUT_PULLUP);
  pinMode(DB4,INPUT_PULLUP);
  pinMode(DB5,INPUT_PULLUP);
  pinMode(DB6,INPUT_PULLUP);
  pinMode(DB7,INPUT_PULLUP);
  pinMode(DB8,INPUT_PULLUP);
  pinMode(DB9,INPUT_PULLUP);
  pinMode(DB10,INPUT_PULLUP);
  pinMode(DB11,INPUT_PULLUP);
  pinMode(DB12,INPUT_PULLUP);
  pinMode(DB13,INPUT_PULLUP);
  pinMode(DB14,INPUT_PULLUP);
  pinMode(DB15,INPUT_PULLUP);

  // Select channel 0
  digitalWrite(CHSEL0,LOW);
  digitalWrite(CHSEL1,LOW);
  digitalWrite(CHSEL2,LOW);

  // Set read range of the ADC to +- 10V, (HIGH), for +- 5V Set low.
  digitalWrite(HW_RNGSEL0, HIGH);

  //Set RD and CS high(default mode)
  digitalWrite(SCLK_RD,HIGH);
  digitalWrite(CS,HIGH);

  // Internal reference, Tied high on board
  digitalWrite(WR_BURST, LOW);
  digitalWrite(SEQEN, LOW); // Enable sequencer, hardware mode only

  // Activate the ADC pin by setting the reset pin high after having had it low
  digitalWrite(RESET, LOW);
  delay(100);
  digitalWrite(RESET, HIGH);
}

void sampleDataUint8(uint8_t* data_array, uint16_t msb_idx, uint16_t lsb_idx)
{
    /*
    Samples a single channel of the ADC and puts the result in the data array.
    */

    // Prefetch all relevant ports once
    
    uint32_t A = GPIOA->IDR;
    uint32_t B = GPIOB->IDR;
    uint32_t D = GPIOD->IDR;
    uint32_t G = GPIOG->IDR;
    uint32_t H = GPIOH->IDR;
    uint32_t I = GPIOI->IDR;
    uint32_t J = GPIOJ->IDR;
    
    // Clear target bytes
    data_array[lsb_idx] = 0;
    data_array[msb_idx] = 0;

    // --- LSB byte ---
    data_array[lsb_idx] =
          READ_PIN(B,15,0)
        | READ_PIN(H,9,1)
        | READ_PIN(B,3,2)
        | READ_PIN(H,11,3)
        | READ_PIN(B,4,4)
        | READ_PIN(H,14,5)
        | READ_PIN(B,14,6)
        | READ_PIN(J,8,7);

    // --- MSB byte ---
    data_array[msb_idx] =
          READ_PIN(D,7,0)
        | READ_PIN(A,11,1)
        | READ_PIN(D,6,2)
        | READ_PIN(G,14,3)
        | READ_PIN(D,3,4)
        | READ_PIN(G,9,5)
        | READ_PIN(B,9,6);

    // Optional flag bit (bit 7 of MSB)
    if (((I >> 2) & 1) == 0) {
        data_array[msb_idx] |= (1 << 7);
    }
}

void select_channel_fast(uint8_t channel_pair) {
  /*
  Selects the target channel on the ADC for sampling
  */
  switch (channel_pair) {
    case 0:
      GPIOI->BSRR = (1 << (7 + 16));
      GPIOI->BSRR = (1 << (5 + 16));
      GPIOC->BSRR = (1 << (13 + 16));
      break;

    case 1:
      GPIOI->BSRR = (1 << 7);
      GPIOI->BSRR = (1 << (5 + 16));
      GPIOC->BSRR = (1 << (13 + 16));
      break;

    case 2:
      GPIOI->BSRR = (1 << (7 + 16));
      GPIOI->BSRR = (1 << 5);
      GPIOC->BSRR = (1 << (13 + 16));
      break;

    case 3:
      GPIOI->BSRR = (1 << 7);
      GPIOI->BSRR = (1 << 5);
      GPIOC->BSRR = (1 << (13 + 16));
      break;

    case 4:
      GPIOI->BSRR = (1 << (7 + 16));
      GPIOI->BSRR = (1 << (5 + 16));
      GPIOC->BSRR = (1 << 13);
      break;

    case 5:
      GPIOI->BSRR = (1 << 7);
      GPIOI->BSRR = (1 << (5 + 16));
      GPIOC->BSRR = (1 << 13);
      break;

    case 6:
      GPIOI->BSRR = (1 << (7 + 16));
      GPIOI->BSRR = (1 << 5);
      GPIOC->BSRR = (1 << 13);
      break;

    case 7:
      GPIOI->BSRR = (1 << 7);
      GPIOI->BSRR = (1 << 5);
      GPIOC->BSRR = (1 << 13);
      break;

    default:
      return;
  }
}


void read_adc_2_uint8_t(uint8_t channel_pair, uint8_t* data_array, uint16_t index1, uint16_t index2){
  /*
  Reads a pair of channels (A and B) on the analog digital converter and puts the result in the data_array
  */
  if (channel_pair>7){
    return;
  }
  // Select the channel to sample during the next cycle(not this one)
  select_channel_fast((channel_pair+1)%8);
  
  GPIOH->BSRR = (1 << 6);
  GPIOH->BSRR = (1 << (6 + 16));
  uint8_t lim = 0;
  while (((GPIOI->IDR & (1 << 6)) >> 6) && lim<200){
    lim+=1;
  }
  
  GPIOI->BSRR = (1 << (4 + 16));
  GPIOH->BSRR = (1 << (12 + 16));
  
  sampleDataUint8(data_array, index1+1, index1);

  GPIOH->BSRR = (1 << 12);
  GPIOI->BSRR = (1 << 4);

  GPIOI->BSRR = (1 << (4 + 16));
  GPIOH->BSRR = (1 << (12 + 16));

  sampleDataUint8(data_array, index2+1, index2);
  GPIOI->BSRR = (1 << 4);
  GPIOH->BSRR = (1 << 12);
}
