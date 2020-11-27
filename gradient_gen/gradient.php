<html>
  
    <head>
        <title>Gradient Generator</title>
    </head>

    <body>
        <form method="post"> 
            <INPUT TYPE = "Text" Size = 20 Value = "<?php echo isset($_POST['colors']) ? $_POST['colors'] : 'ffff00,0000ff' ?>" Name = "colors">
            <select id="steps" name="steps" value="256">
              <option value="32">32</option>
              <option value="128">128</option>
              <option value="256">256</option>
            </select>
            <input type="submit" name="generate"
                    class="button" value="Generate" />
            Color Picker:
            <input type="color" name="color">
        </form>
        <?php
        include("palette.php");
        $myPalette=new palette(explode(',', $_POST['colors']), $_POST['steps']);

        echo $myPalette->printColors();
        echo $myPalette->printCPP();
        echo $myPalette->printTest();
        ?>
    </body>

</html>
