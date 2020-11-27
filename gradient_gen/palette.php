<?php
/*
Vladimir Guzmán
http://www.maintask.com
-----------------------
Este código es de uso absolutamente libre.
-----------------------
Adaptación del código de steve@slayeroffice.com
http://slayeroffice.com/tools/color_palette/
Basado a su vez en una idea de Andy Clarke:
http://www.stuffandnonsense.co.uk/archives/creating_colour_palettes.html
*/
class palette{

  protected $colors=array();  //Arreglo de colores por los cuales debe pasar la paleta
  public $colorPath=array();  //Arreglo de colores finales de la paleta
  protected $numSteps=10;

  public function __construct($colors=NULL,$numSteps=NULL){
    if($colors!=NULL) $this->colors=$colors;
    if($numSteps!=NULL) $this->numSteps=$numSteps;
    $this->generate();
  }

  public function generate(){
    if(sizeof($this->colors)<2) return(FALSE);
    $steps=floor($this->numSteps/(sizeof($this->colors)-1));
    $steps=ceil(($this->numSteps-sizeof($this->colors))/(sizeof($this->colors)-1))+1;
    for($i=0;$i<sizeof($this->colors)-1;$i++){
      $this->fade($this->colors[$i],$this->colors[$i+1],$steps);
    }
  }

  private function fade($from,$to,$steps){
    $from=$this->longHexToDec($from);
    if(sizeof($this->colorPath)==0) array_push($this->colorPath,$this->decToLongHex($from));
    $to=$this->longHexToDec($to);
    for($i=1;$i<$steps;$i++){
      $nColor=$this->setColorHue($from,$i/$steps,$to);
      if(sizeof($this->colorPath)<$this->numSteps) array_push($this->colorPath,$this->decToLongHex($nColor));
    }
    if(sizeof($this->colorPath)<$this->numSteps) array_push($this->colorPath,$this->decToLongHex($to));
  }

  private function longHexToDec($hex){
    $r=hexdec(substr($hex,0,2));
    $g=hexdec(substr($hex,2,2));
    $b=hexdec(substr($hex,4,2));
    return(array($r,$g,$b));
  }

  private function decToLongHex($rgb){
    $r = str_pad(dechex($rgb[0]), 2, '0', STR_PAD_LEFT);
    $g = str_pad(dechex($rgb[1]), 2, '0', STR_PAD_LEFT);
    $b = str_pad(dechex($rgb[2]), 2, '0', STR_PAD_LEFT);
    return($r . $g . $b);
  }

  private function setColorHue($originColor,$opacityPercent,$maskRGB) {
    $returnColor=array();
    for($w=0;$w<sizeof($originColor);$w++) $returnColor[$w] = floor($originColor[$w]*(1.0-$opacityPercent)) + round($maskRGB[$w]*($opacityPercent));
    return $returnColor;
  }

  public function printColors(){
    $string="<table border=\"1\">\n\t<tr>\n";
    for($i=0;$i<sizeof($this->colors);$i++){
      $string.="\t\t<td bgcolor=\"#" . $this->colors[$i] . "\">" . $this->colors[$i] . "</td>\n";
    }
    $string.="\t</tr>\n</table>\n";
    return($string);
  }

  public function printTest(){
    $string="<table border=\"1\">\n";
    for($i=0;$i<sizeof($this->colorPath);$i++){
      $string.="\t\t<tr><td bgcolor=\"#" . $this->colorPath[$i] . "\">" . $this->colorPath[$i] . "</td></tr>\n";
    }
    $string.="\t</table>\n";
    return($string);
  }

  public function printCPP(){
    $string="    {";
    for($i=0;$i<sizeof($this->colorPath);$i++){
      $string.="0x" . strtoupper($this->colorPath[$i]) . ", ";
    }
    $string=substr($string, 0, -2) . "},\n";
    return($string);
  }
}
?>
