#ifndef SECP256K1H
#define SECP256K1H

#include "Point.h"
#include <string>
#include <vector>

// Address type
#define P2PKH  0
#define P2SH   1
#define BECH32 2

#define NUM_GTABLE_CHUNK 16    //number of GTable chunks that are pre-computed and stored in memory
#define NUM_GTABLE_VALUE 65536 //number of GTable values per chunk (all possible states) (2 ^ (bits_per_chunk))

class Secp256K1 {

public:

  Secp256K1();
  ~Secp256K1();
  void Init();
  Point ComputePublicKey(Int *privKey);
  Point NextKey(Point &key);
  bool  EC(Point &p);

  Point Add(Point &p1, Point &p2);
  Point AddPoint2(Point &p1, Point &p2);  // Renomeado de Add2 para AddPoint2
  Point AddDirect(Point &p1, Point &p2);
  Point Double(Point &p);
  Point DoubleDirect(Point &p);

  Point G;                 // Generator
  Int   order;             // Curve order

  Point GTable[NUM_GTABLE_CHUNK * NUM_GTABLE_VALUE]; // Generator table

private:

  uint8_t GetByte(std::string &str,int idx);

  Int GetY(Int x, bool isEven);
  

};

#endif // SECP256K1H
