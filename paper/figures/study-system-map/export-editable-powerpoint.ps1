param(
    [string]$BackgroundPath = (Join-Path $PSScriptRoot "study-system-map-editable-background.jpg"),
    [string]$AreaDonutPath = (Join-Path $PSScriptRoot "study-system-map-editable-area-donut.svg"),
    [string]$DischargeDonutPath = (Join-Path $PSScriptRoot "study-system-map-editable-discharge-donut.svg"),
    [string]$NorthArrowPath = (Join-Path $PSScriptRoot "data\north-arrow-white.svg"),
    [string]$MetadataPath = (Join-Path $PSScriptRoot "data\study-system-metadata.json"),
    [string]$OutputPath = (Join-Path $PSScriptRoot "study-system-map.pptx")
)

$ErrorActionPreference = "Stop"

$output = [System.IO.Path]::GetFullPath($OutputPath)
if (Test-Path -LiteralPath $output) {
    throw "Refusing to overwrite existing PowerPoint: $output. Use -OutputPath with a new filename to preserve manual edits."
}

function ConvertTo-OfficeRgb {
    param(
        [int]$Red,
        [int]$Green,
        [int]$Blue
    )
    return $Red + (256 * $Green) + (65536 * $Blue)
}

function Add-MapLabel {
    param(
        [object]$Slide,
        [string]$Name,
        [string]$Text,
        [double]$LeftFraction,
        [double]$TopFraction,
        [double]$WidthFraction,
        [double]$HeightFraction,
        [double]$SlideWidth,
        [double]$SlideHeight,
        [double]$FontSize,
        [int]$Color,
        [bool]$Bold = $false,
        [double]$Rotation = 0.0,
        [int]$Alignment = 2,
        [bool]$Italic = $false,
        [bool]$DarkGlow = $false
    )

    $shape = $Slide.Shapes.AddTextbox(
        1,
        $LeftFraction * $SlideWidth,
        $TopFraction * $SlideHeight,
        $WidthFraction * $SlideWidth,
        $HeightFraction * $SlideHeight
    )
    $shape.Name = "Label - $Name"
    $shape.Rotation = $Rotation
    $shape.Fill.Visible = 0
    $shape.Line.Visible = 0
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    $shape.TextFrame.WordWrap = -1
    $shape.TextFrame.VerticalAnchor = 3
    $shape.TextFrame.TextRange.Text = $Text.Replace("`n", "`r")
    $shape.TextFrame.TextRange.ParagraphFormat.Alignment = $Alignment
    $shape.TextFrame.TextRange.Font.Name = "Inter"
    $shape.TextFrame.TextRange.Font.Size = $FontSize
    $shape.TextFrame.TextRange.Font.Bold = [int]$Bold * -1
    $shape.TextFrame.TextRange.Font.Italic = [int]$Italic * -1
    $shape.TextFrame.TextRange.Font.Color.RGB = $Color

    try {
        $shape.TextFrame2.TextRange.Font.Name = "Inter"
        $shape.TextFrame2.TextRange.Font.Size = $FontSize
        $shape.TextFrame2.TextRange.Font.Bold = [int]$Bold * -1
        $shape.TextFrame2.TextRange.Font.Italic = [int]$Italic * -1
        $shape.TextFrame2.TextRange.Font.Fill.Visible = -1
        $shape.TextFrame2.TextRange.Font.Fill.Solid()
        $shape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $Color
        $shape.TextFrame2.TextRange.Font.Fill.Transparency = 0
        $shape.TextFrame2.TextRange.Font.Line.Visible = -1
        if ($DarkGlow) {
            $shape.TextFrame2.TextRange.Font.Line.Visible = 0
            $shape.TextFrame2.TextRange.Font.Glow.Color.RGB = ConvertTo-OfficeRgb 17 24 39
            $shape.TextFrame2.TextRange.Font.Glow.Radius = 2.5
            $shape.TextFrame2.TextRange.Font.Glow.Transparency = 0.10
        }
        else {
            $shape.TextFrame2.TextRange.Font.Line.ForeColor.RGB = ConvertTo-OfficeRgb 17 24 39
            $shape.TextFrame2.TextRange.Font.Line.Transparency = 0.12
            $shape.TextFrame2.TextRange.Font.Line.Weight = 0.45
        }
    }
    catch {
        # Older PowerPoint versions may not expose the TextFrame2 font outline.
    }

    return $shape
}

function Add-PlainText {
    param(
        [object]$Slide,
        [string]$Name,
        [string]$Text,
        [double]$LeftFraction,
        [double]$TopFraction,
        [double]$WidthFraction,
        [double]$HeightFraction,
        [double]$SlideWidth,
        [double]$SlideHeight,
        [double]$FontSize,
        [int]$Color,
        [bool]$Bold = $false,
        [int]$Alignment = 2,
        [bool]$Italic = $false
    )

    $shape = $Slide.Shapes.AddTextbox(
        1,
        $LeftFraction * $SlideWidth,
        $TopFraction * $SlideHeight,
        $WidthFraction * $SlideWidth,
        $HeightFraction * $SlideHeight
    )
    $shape.Name = $Name
    $shape.Fill.Visible = 0
    $shape.Line.Visible = 0
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    $shape.TextFrame.WordWrap = -1
    $shape.TextFrame.VerticalAnchor = 3
    $shape.TextFrame.TextRange.Text = $Text.Replace("`n", "`r")
    $shape.TextFrame.TextRange.ParagraphFormat.Alignment = $Alignment
    $shape.TextFrame.TextRange.Font.Name = "Inter"
    $shape.TextFrame.TextRange.Font.Size = $FontSize
    $shape.TextFrame.TextRange.Font.Bold = [int]$Bold * -1
    $shape.TextFrame.TextRange.Font.Italic = [int]$Italic * -1
    $shape.TextFrame.TextRange.Font.Color.RGB = $Color

    try {
        $shape.TextFrame2.TextRange.Font.Name = "Inter"
        $shape.TextFrame2.TextRange.Font.Size = $FontSize
        $shape.TextFrame2.TextRange.Font.Bold = [int]$Bold * -1
        $shape.TextFrame2.TextRange.Font.Italic = [int]$Italic * -1
        $shape.TextFrame2.TextRange.Font.Fill.Visible = -1
        $shape.TextFrame2.TextRange.Font.Fill.Solid()
        $shape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $Color
        $shape.TextFrame2.TextRange.Font.Line.Visible = 0
    }
    catch {
        # The legacy TextFrame settings above remain sufficient.
    }

    return $shape
}

function Add-Leader {
    param(
        [object]$Slide,
        [string]$Name,
        [double]$X1Fraction,
        [double]$Y1Fraction,
        [double]$X2Fraction,
        [double]$Y2Fraction,
        [double]$SlideWidth,
        [double]$SlideHeight,
        [int]$Color
    )

    $line = $Slide.Shapes.AddLine(
        $X1Fraction * $SlideWidth,
        $Y1Fraction * $SlideHeight,
        $X2Fraction * $SlideWidth,
        $Y2Fraction * $SlideHeight
    )
    $line.Name = "Leader - $Name"
    $line.Line.ForeColor.RGB = $Color
    $line.Line.Weight = 1.2
    return $line
}

function Group-NamedShapes {
    param(
        [object]$Slide,
        [string[]]$Names,
        [string]$GroupName
    )

    try {
        $range = $Slide.Shapes.Range([object[]]$Names)
        $group = $range.Group()
        $group.Name = $GroupName
        return $group
    }
    catch {
        Write-Warning "Could not group '$GroupName'; its named parts remain editable."
        return $null
    }
}

$background = (Resolve-Path -LiteralPath $BackgroundPath).Path
$areaDonut = (Resolve-Path -LiteralPath $AreaDonutPath).Path
$dischargeDonut = (Resolve-Path -LiteralPath $DischargeDonutPath).Path
$northArrow = (Resolve-Path -LiteralPath $NorthArrowPath).Path
$metadata = Get-Content -LiteralPath $MetadataPath -Raw | ConvertFrom-Json
$areaShareLabel = "{0:N1}%" -f [double]$metadata.animas_plus_navajo_area_share_pct
$dischargeShareLabel = "{0:N1}%" -f [double]$metadata.bluff_headwater_contribution.volume_share_pct
$powerPoint = $null
$presentation = $null
$slide = $null

try {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    $powerPoint.Visible = -1
    $presentation = $powerPoint.Presentations.Add()

    # Match the generated map's 2866:2086 aspect ratio.
    $slideHeight = 720.0
    $slideWidth = $slideHeight * 2866.0 / 2086.0
    $presentation.PageSetup.SlideWidth = $slideWidth
    $presentation.PageSetup.SlideHeight = $slideHeight

    $slide = $presentation.Slides.Add(1, 12)
    $backgroundShape = $slide.Shapes.AddPicture(
        $background,
        0,
        -1,
        0,
        0,
        $slideWidth,
        $slideHeight
    )
    $backgroundShape.Name = "Fixed map background"

    $white = ConvertTo-OfficeRgb 255 255 255
    $yellow = ConvertTo-OfficeRgb 250 204 21
    $riverBlue = ConvertTo-OfficeRgb 147 197 253
    $gage = ConvertTo-OfficeRgb 251 113 133
    $textColor = ConvertTo-OfficeRgb 17 24 39
    $muted = ConvertTo-OfficeRgb 100 116 139
    $panelLine = ConvertTo-OfficeRgb 203 213 225
    $otherBasins = ConvertTo-OfficeRgb 203 213 225
    $mainstemBlue = ConvertTo-OfficeRgb 43 108 176

    # Positions reproduce the manually refined PowerPoint layout.
    Add-MapLabel $slide "Lake Powell" "Lake Powell" -0.008829 0.312658 0.099807 0.070684 $slideWidth $slideHeight 21.0 $white $true 0 2 $true | Out-Null
    Add-MapLabel $slide "San Juan" "San Juan" 0.267613 0.278829 0.184995 0.035342 $slideWidth $slideHeight 21.0 $riverBlue $true 8 2 $true | Out-Null
    Add-MapLabel $slide "River" "River" 0.377853 0.307642 0.089332 0.035342 $slideWidth $slideHeight 21.0 $riverBlue $true 34.0706 2 $true | Out-Null
    Add-MapLabel $slide "Animas River basin" "Animas`nRiver`nbasin" 0.653710 0.169658 0.104997 0.100977 $slideWidth $slideHeight 20.0 $yellow $true 0 2 $false | Out-Null
    Add-MapLabel $slide "Navajo Reservoir basin" "Navajo`nReservoir`nbasin" 0.780878 0.225658 0.124997 0.100977 $slideWidth $slideHeight 20.0 $yellow $true 0 2 $false | Out-Null
    Add-MapLabel $slide "Navajo Reservoir" "Navajo Reservoir" 0.788700 0.417488 0.081777 0.067318 $slideWidth $slideHeight 20.0 $white $true 0 2 $true | Out-Null

    Add-MapLabel $slide "SJ nr Bluff" "SJ nr Bluff" 0.137221 0.362520 0.153180 0.030293 $slideWidth $slideHeight 18.0 $gage $false 0 2 $true $true | Out-Null
    Add-MapLabel $slide "SJ at Four Corners" "SJ at Four Corners" 0.285212 0.410853 0.153180 0.030293 $slideWidth $slideHeight 18.0 $gage $false 0 2 $true $true | Out-Null
    Add-MapLabel $slide "SJ at Shiprock" "SJ at Shiprock" 0.380072 0.498187 0.153180 0.030293 $slideWidth $slideHeight 18.0 $gage $false 0 2 $true $true | Out-Null
    Add-MapLabel $slide "Animas at Farmington" "Animas at`nFarmington" 0.495553 0.399707 0.153180 0.060586 $slideWidth $slideHeight 18.0 $gage $false 0 2 $true $true | Out-Null
    Add-MapLabel $slide "SJ at Farmington" "SJ at Farmington" 0.479541 0.551853 0.153180 0.030293 $slideWidth $slideHeight 18.0 $gage $false 0 2 $true $true | Out-Null
    Add-MapLabel $slide "SJ nr Archuleta" "SJ nr`nArchuleta" 0.689155 0.498707 0.109268 0.060586 $slideWidth $slideHeight 18.0 $gage $false 0 2 $true $true | Out-Null

    Add-Leader $slide "Animas at Farmington" 0.617195 0.516000 0.579348 0.461000 $slideWidth $slideHeight $gage | Out-Null
    Add-Leader $slide "SJ at Farmington" 0.585984 0.552000 0.613983 0.522000 $slideWidth $slideHeight $gage | Out-Null

    # Editable four-row map legend.
    $legendNames = [System.Collections.Generic.List[string]]::new()
    $legendPanel = $slide.Shapes.AddShape(5, 0.025 * $slideWidth, 0.028 * $slideHeight, 0.235 * $slideWidth, 0.140 * $slideHeight)
    $legendPanel.Name = "Legend - panel"
    $legendPanel.Fill.Solid()
    $legendPanel.Fill.ForeColor.RGB = $white
    $legendPanel.Fill.Transparency = 0.08
    $legendPanel.Line.ForeColor.RGB = $panelLine
    $legendPanel.Line.Weight = 0.8
    $legendNames.Add($legendPanel.Name)

    $legendRowY = @(0.050, 0.080, 0.110, 0.141)
    $legendHeadwater = $slide.Shapes.AddLine(0.034 * $slideWidth, $legendRowY[0] * $slideHeight, 0.064 * $slideWidth, $legendRowY[0] * $slideHeight)
    $legendHeadwater.Name = "Legend - Animas and Navajo line"
    $legendHeadwater.Line.ForeColor.RGB = $yellow
    $legendHeadwater.Line.Weight = 2.2
    $legendNames.Add($legendHeadwater.Name)
    $legendNames.Add((Add-PlainText $slide "Legend - Animas and Navajo text" "Animas + Navajo basins" 0.073 0.038 0.178 0.024 $slideWidth $slideHeight 17.0 $textColor $false 1).Name)

    $legendBasinUnder = $slide.Shapes.AddLine(0.034 * $slideWidth, $legendRowY[1] * $slideHeight, 0.064 * $slideWidth, $legendRowY[1] * $slideHeight)
    $legendBasinUnder.Name = "Legend - basin boundary underlay"
    $legendBasinUnder.Line.ForeColor.RGB = $muted
    $legendBasinUnder.Line.Weight = 3.0
    $legendNames.Add($legendBasinUnder.Name)
    $legendBasin = $slide.Shapes.AddLine(0.034 * $slideWidth, $legendRowY[1] * $slideHeight, 0.064 * $slideWidth, $legendRowY[1] * $slideHeight)
    $legendBasin.Name = "Legend - basin boundary"
    $legendBasin.Line.ForeColor.RGB = $white
    $legendBasin.Line.Weight = 1.4
    $legendNames.Add($legendBasin.Name)
    $legendNames.Add((Add-PlainText $slide "Legend - basin text" "San Juan basin" 0.073 0.068 0.178 0.024 $slideWidth $slideHeight 17.0 $textColor $false 1).Name)

    $legendMainstem = $slide.Shapes.AddLine(0.034 * $slideWidth, $legendRowY[2] * $slideHeight, 0.064 * $slideWidth, $legendRowY[2] * $slideHeight)
    $legendMainstem.Name = "Legend - mainstem line"
    $legendMainstem.Line.ForeColor.RGB = $mainstemBlue
    $legendMainstem.Line.Weight = 2.2
    $legendNames.Add($legendMainstem.Name)
    $legendNames.Add((Add-PlainText $slide "Legend - mainstem text" "San Juan mainstem" 0.073 0.098 0.178 0.024 $slideWidth $slideHeight 17.0 $textColor $false 1).Name)

    $gageSize = 10.0
    $legendGage = $slide.Shapes.AddShape(9, 0.049 * $slideWidth - $gageSize / 2.0, $legendRowY[3] * $slideHeight - $gageSize / 2.0, $gageSize, $gageSize)
    $legendGage.Name = "Legend - gage marker"
    $legendGage.Fill.Solid()
    $legendGage.Fill.ForeColor.RGB = $gage
    $legendGage.Line.ForeColor.RGB = $textColor
    $legendGage.Line.Weight = 1.0
    $legendNames.Add($legendGage.Name)
    $legendNames.Add((Add-PlainText $slide "Legend - gage text" "USGS gage" 0.073 0.129 0.178 0.024 $slideWidth $slideHeight 17.0 $textColor $false 1).Name)
    Group-NamedShapes $slide $legendNames.ToArray() "Map legend" | Out-Null

    # Editable north arrow and scale bar.
    $northShape = $slide.Shapes.AddPicture($northArrow, 0, -1, 0.027 * $slideWidth, 0.841 * $slideHeight, 0.058 * $slideWidth, 0.092 * $slideHeight)
    $northShape.Name = "North arrow"

    $scaleNames = [System.Collections.Generic.List[string]]::new()
    $scaleUnder = $slide.Shapes.AddLine(0.045 * $slideWidth, 0.957 * $slideHeight, 0.286 * $slideWidth, 0.957 * $slideHeight)
    $scaleUnder.Name = "Scale bar - dark underlay"
    $scaleUnder.Line.ForeColor.RGB = $textColor
    $scaleUnder.Line.Weight = 6.0
    $scaleNames.Add($scaleUnder.Name)
    $scaleLine = $slide.Shapes.AddLine(0.045 * $slideWidth, 0.957 * $slideHeight, 0.286 * $slideWidth, 0.957 * $slideHeight)
    $scaleLine.Name = "Scale bar - white line"
    $scaleLine.Line.ForeColor.RGB = $white
    $scaleLine.Line.Weight = 4.0
    $scaleNames.Add($scaleLine.Name)
    $scaleNames.Add((Add-MapLabel $slide "Scale bar - label" "100 km" 0.116 0.916 0.100 0.030 $slideWidth $slideHeight 18.0 $white $true 0 2 $false $false).Name)
    Group-NamedShapes $slide $scaleNames.ToArray() "Scale bar" | Out-Null

    # Fully editable inset; the two rings are independent vector SVG objects.
    $insetNames = [System.Collections.Generic.List[string]]::new()
    $insetPanel = $slide.Shapes.AddShape(1, 0.590 * $slideWidth, 0.660 * $slideHeight, 0.370 * $slideWidth, 0.300 * $slideHeight)
    $insetPanel.Name = "Inset - panel"
    $insetPanel.Fill.Solid()
    $insetPanel.Fill.ForeColor.RGB = $white
    $insetPanel.Line.Visible = 0
    $insetNames.Add($insetPanel.Name)
    $insetNames.Add((Add-PlainText $slide "Inset - title" "Disproportionate headwater contribution" 0.600 0.666 0.350 0.030 $slideWidth $slideHeight 18.0 $textColor $true 2).Name)

    $insetLegendSquareSize = 8.0
    $insetGoldSquare = $slide.Shapes.AddShape(1, 0.628 * $slideWidth, 0.706 * $slideHeight, $insetLegendSquareSize, $insetLegendSquareSize)
    $insetGoldSquare.Name = "Inset - headwater legend swatch"
    $insetGoldSquare.Fill.Solid()
    $insetGoldSquare.Fill.ForeColor.RGB = $yellow
    $insetGoldSquare.Line.Visible = 0
    $insetNames.Add($insetGoldSquare.Name)
    $insetNames.Add((Add-PlainText $slide "Inset - headwater legend text" "Animas + Navajo" 0.648 0.696 0.140 0.024 $slideWidth $slideHeight 14.0 $textColor $false 1).Name)
    $insetGraySquare = $slide.Shapes.AddShape(1, 0.800 * $slideWidth, 0.706 * $slideHeight, $insetLegendSquareSize, $insetLegendSquareSize)
    $insetGraySquare.Name = "Inset - other basins legend swatch"
    $insetGraySquare.Fill.Solid()
    $insetGraySquare.Fill.ForeColor.RGB = $otherBasins
    $insetGraySquare.Line.Visible = 0
    $insetNames.Add($insetGraySquare.Name)
    $insetNames.Add((Add-PlainText $slide "Inset - other basins legend text" "Rest of San Juan basin" 0.820 0.696 0.135 0.024 $slideWidth $slideHeight 14.0 $textColor $false 1).Name)

    $areaDonutShape = $slide.Shapes.AddPicture($areaDonut, 0, -1, 0.602 * $slideWidth, 0.735 * $slideHeight, 0.125 * $slideWidth, 0.155 * $slideHeight)
    $areaDonutShape.Name = "Inset - area donut"
    $insetNames.Add($areaDonutShape.Name)
    $dischargeDonutShape = $slide.Shapes.AddPicture($dischargeDonut, 0, -1, 0.822 * $slideWidth, 0.735 * $slideHeight, 0.125 * $slideWidth, 0.155 * $slideHeight)
    $dischargeDonutShape.Name = "Inset - discharge donut"
    $insetNames.Add($dischargeDonutShape.Name)
    $insetNames.Add((Add-PlainText $slide "Inset - area percentage" $areaShareLabel 0.602 0.795 0.125 0.040 $slideWidth $slideHeight 19.0 $textColor $true 2).Name)
    $insetNames.Add((Add-PlainText $slide "Inset - discharge percentage" $dischargeShareLabel 0.822 0.795 0.125 0.040 $slideWidth $slideHeight 19.0 $textColor $true 2).Name)
    $insetNames.Add((Add-PlainText $slide "Inset - connector text" "accounts for" 0.730 0.783 0.090 0.028 $slideWidth $slideHeight 14.0 $textColor $true 2).Name)
    $insetArrow = $slide.Shapes.AddLine(0.742 * $slideWidth, 0.824 * $slideHeight, 0.808 * $slideWidth, 0.824 * $slideHeight)
    $insetArrow.Name = "Inset - connector arrow"
    $insetArrow.Line.ForeColor.RGB = $muted
    $insetArrow.Line.Weight = 1.4
    $insetArrow.Line.EndArrowheadStyle = 4
    $insetNames.Add($insetArrow.Name)
    $insetNames.Add((Add-PlainText $slide "Inset - area description" "Share of total`nSan Juan basin area" 0.595 0.900 0.145 0.048 $slideWidth $slideHeight 14.0 $textColor $false 2).Name)
    $insetNames.Add((Add-PlainText $slide "Inset - discharge description" "Contribution to`ndischarge at`nSJ nr Bluff" 0.812 0.897 0.145 0.055 $slideWidth $slideHeight 14.0 $textColor $false 2).Name)
    Group-NamedShapes $slide $insetNames.ToArray() "Headwater contribution inset" | Out-Null

    Add-MapLabel $slide "Data attribution" "Imagery: USDA, USGS The National Map`nHydrography contains information from MERIT Hydro v1.0.1 (ODbL 1.0)" 0.622 0.946 0.370 0.044 $slideWidth $slideHeight 8.3 $white $false 0 3 $false $false | Out-Null

    $presentation.SaveAs($output, 24)
}
finally {
    if ($presentation -ne $null) {
        try {
            $presentation.Close()
        }
        catch {
            # SaveAs can release the COM presentation before cleanup completes.
        }
    }
    if ($powerPoint -ne $null) {
        try {
            $powerPoint.Quit()
        }
        catch {
            # PowerPoint may already have exited after releasing the presentation.
        }
    }
    if ($slide -ne $null) {
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($slide)
    }
    if ($presentation -ne $null) {
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($presentation)
    }
    if ($powerPoint -ne $null) {
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($powerPoint)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

Write-Output "Wrote $output"
