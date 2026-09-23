param(
    [string]$OutputPath = (Join-Path $PSScriptRoot "objective-locations-schematic-editable.pptx")
)

$ErrorActionPreference = "Stop"

function ConvertTo-OfficeRgb {
    param([string]$Hex)
    $value = $Hex.TrimStart('#')
    $r = [Convert]::ToInt32($value.Substring(0, 2), 16)
    $g = [Convert]::ToInt32($value.Substring(2, 2), 16)
    $b = [Convert]::ToInt32($value.Substring(4, 2), 16)
    return $r + (256 * $g) + (65536 * $b)
}

function Add-EditableText {
    param(
        [object]$Slide,
        [string]$Name,
        [string]$Text,
        [double]$X,
        [double]$Y,
        [double]$W,
        [double]$H,
        [double]$SlideWidth,
        [double]$SlideHeight,
        [double]$FontSize,
        [int]$Color,
        [bool]$Bold = $false,
        [int]$Alignment = 2,
        [int]$VerticalAnchor = 3
    )

    $shape = $Slide.Shapes.AddTextbox(
        1,
        $X * $SlideWidth,
        (1.0 - $Y - $H) * $SlideHeight,
        $W * $SlideWidth,
        $H * $SlideHeight
    )
    $shape.Name = $Name
    $shape.Fill.Visible = 0
    $shape.Line.Visible = 0
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    $shape.TextFrame.WordWrap = -1
    $shape.TextFrame.AutoSize = 0
    $shape.TextFrame.VerticalAnchor = $VerticalAnchor
    $shape.TextFrame.TextRange.Text = $Text.Replace("`n", "`r")
    $shape.TextFrame.TextRange.ParagraphFormat.Alignment = $Alignment
    $shape.TextFrame.TextRange.ParagraphFormat.SpaceAfter = 0
    $shape.TextFrame.TextRange.Font.Name = "Inter"
    $shape.TextFrame.TextRange.Font.Size = $FontSize
    $shape.TextFrame.TextRange.Font.Bold = [int]$Bold * -1
    $shape.TextFrame.TextRange.Font.Color.RGB = $Color
    try {
        $shape.TextFrame2.TextRange.Font.Name = "Inter"
        $shape.TextFrame2.TextRange.Font.Size = $FontSize
        $shape.TextFrame2.TextRange.Font.Bold = [int]$Bold * -1
        $shape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $Color
        $shape.TextFrame2.TextRange.Font.Line.Visible = 0
        $shape.TextFrame2.VerticalAnchor = $VerticalAnchor
    }
    catch {
        # Legacy TextFrame formatting remains sufficient.
    }
    return $shape
}

function Add-EditableLine {
    param(
        [object]$Slide,
        [string]$Name,
        [double]$X1,
        [double]$Y1,
        [double]$X2,
        [double]$Y2,
        [double]$SlideWidth,
        [double]$SlideHeight,
        [int]$Color,
        [double]$Weight = 1.5,
        [bool]$Arrow = $false
    )

    $shape = $Slide.Shapes.AddLine(
        $X1 * $SlideWidth,
        (1.0 - $Y1) * $SlideHeight,
        $X2 * $SlideWidth,
        (1.0 - $Y2) * $SlideHeight
    )
    $shape.Name = $Name
    $shape.Line.ForeColor.RGB = $Color
    $shape.Line.Weight = $Weight
    if ($Arrow) {
        $shape.Line.EndArrowheadStyle = 4
    }
    return $shape
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
        Write-Warning "Could not group '$GroupName'; its parts remain editable."
        return $null
    }
}

function Add-EditableConnector {
    param(
        [object]$Slide,
        [string]$Name,
        [object[]]$Points,
        [double]$SlideWidth,
        [double]$SlideHeight,
        [int]$Color
    )

    $names = [System.Collections.Generic.List[string]]::new()
    for ($i = 0; $i -lt $Points.Count - 1; $i++) {
        $start = $Points[$i]
        $end = $Points[$i + 1]
        $lineArgs = @{
            Slide       = $Slide
            Name        = "$Name - segment $($i + 1)"
            X1          = [double]$start[0]
            Y1          = [double]$start[1]
            X2          = [double]$end[0]
            Y2          = [double]$end[1]
            SlideWidth  = $SlideWidth
            SlideHeight = $SlideHeight
            Color       = $Color
            Weight      = 1.5
            Arrow       = ($i -eq $Points.Count - 2)
        }
        $segment = Add-EditableLine @lineArgs
        $names.Add($segment.Name)
    }
    Group-NamedShapes $Slide $names.ToArray() $Name | Out-Null
}

function Add-ObjectiveCard {
    param(
        [object]$Slide,
        [string]$Name,
        [string]$Title,
        [string[]]$Lines,
        [string]$Badge,
        [double]$X,
        [double]$Y,
        [double]$W,
        [double]$H,
        [double]$SlideWidth,
        [double]$SlideHeight,
        [int]$Color,
        [int]$TextColor,
        [int]$MutedColor,
        [bool]$Dashed = $false,
        [double]$BodyFontSize = 11.5,
        [double]$TitleFontSize = 14.0
    )

    $names = [System.Collections.Generic.List[string]]::new()
    $card = $Slide.Shapes.AddShape(
        5,
        $X * $SlideWidth,
        (1.0 - $Y - $H) * $SlideHeight,
        $W * $SlideWidth,
        $H * $SlideHeight
    )
    $card.Name = "$Name - body"
    $card.Fill.Solid()
    $card.Fill.ForeColor.RGB = ConvertTo-OfficeRgb "#FFFFFF"
    $card.Line.ForeColor.RGB = $Color
    $card.Line.Weight = 1.6
    if ($Dashed) {
        $card.Line.DashStyle = 4
    }
    $names.Add($card.Name)

    $headerHeight = 0.039
    $header = $Slide.Shapes.AddShape(
        5,
        $X * $SlideWidth,
        (1.0 - ($Y + $H)) * $SlideHeight,
        $W * $SlideWidth,
        $headerHeight * $SlideHeight
    )
    $header.Name = "$Name - header"
    $header.Fill.Solid()
    $header.Fill.ForeColor.RGB = $Color
    $header.Line.Visible = 0
    $names.Add($header.Name)

    $headerSquare = $Slide.Shapes.AddShape(
        1,
        $X * $SlideWidth,
        (1.0 - ($Y + $H) + $headerHeight * 0.48) * $SlideHeight,
        $W * $SlideWidth,
        ($headerHeight * 0.52) * $SlideHeight
    )
    $headerSquare.Name = "$Name - header lower edge"
    $headerSquare.Fill.Solid()
    $headerSquare.Fill.ForeColor.RGB = $Color
    $headerSquare.Line.Visible = 0
    $names.Add($headerSquare.Name)

    $titleArgs = @{
        Slide       = $Slide
        Name        = "$Name - title"
        Text        = $Title
        X           = $X + 0.012
        Y           = $Y + $H - $headerHeight + 0.003
        W           = $W - 0.024
        H           = $headerHeight - 0.006
        SlideWidth  = $SlideWidth
        SlideHeight = $SlideHeight
        FontSize    = $TitleFontSize
        Color       = ConvertTo-OfficeRgb "#FFFFFF"
        Bold        = $true
        Alignment   = 1
    }
    $titleShape = Add-EditableText @titleArgs
    $names.Add($titleShape.Name)

    $badgeWidth = if ($Badge -eq "Model-enabling") { 0.105 } else { 0.080 }
    $badgeHeight = 0.028
    $badgeX = $X + $W - $badgeWidth - 0.006
    $badgeY = $Y + $H + 0.006
    $badgeShape = $Slide.Shapes.AddShape(
        5,
        $badgeX * $SlideWidth,
        (1.0 - $badgeY - $badgeHeight) * $SlideHeight,
        $badgeWidth * $SlideWidth,
        $badgeHeight * $SlideHeight
    )
    $badgeShape.Name = "$Name - priority badge"
    $badgeShape.Fill.Solid()
    $badgeShape.Fill.ForeColor.RGB = ConvertTo-OfficeRgb "#FFFFFF"
    $badgeShape.Line.ForeColor.RGB = $Color
    $badgeShape.Line.Weight = 0.9
    $names.Add($badgeShape.Name)

    $badgeArgs = @{
        Slide       = $Slide
        Name        = "$Name - priority text"
        Text        = $Badge
        X           = $badgeX
        Y           = $badgeY
        W           = $badgeWidth
        H           = $badgeHeight
        SlideWidth  = $SlideWidth
        SlideHeight = $SlideHeight
        FontSize    = 10.0
        Color       = $Color
        Bold        = $true
        Alignment   = 2
    }
    $badgeText = Add-EditableText @badgeArgs
    $names.Add($badgeText.Name)

    $bodyText = $Lines -join "`n"
    $bodyArgs = @{
        Slide       = $Slide
        Name        = "$Name - criterion text"
        Text        = $bodyText
        X           = $X + 0.010
        Y           = $Y + 0.010
        W           = $W - 0.020
        H           = $H - $headerHeight - 0.018
        SlideWidth  = $SlideWidth
        SlideHeight = $SlideHeight
        FontSize    = $BodyFontSize
        Color       = $TextColor
        Bold        = $false
        Alignment   = 2
        VerticalAnchor = 1
    }
    $bodyShape = Add-EditableText @bodyArgs
    try {
        $bodyShape.TextFrame.TextRange.Paragraphs(1, 1).Font.Bold = -1
        $bodyShape.TextFrame.TextRange.Paragraphs($Lines.Count, 1).Font.Color.RGB = $MutedColor
    }
    catch {
        # Uniform body formatting is an acceptable fallback.
    }
    $names.Add($bodyShape.Name)

    Group-NamedShapes $Slide $names.ToArray() $Name | Out-Null
}

$output = [System.IO.Path]::GetFullPath($OutputPath)
if (Test-Path -LiteralPath $output) {
    Remove-Item -LiteralPath $output -Force
}

$textColor = ConvertTo-OfficeRgb "#111827"
$muted = ConvertTo-OfficeRgb "#64748B"
$context = ConvertTo-OfficeRgb "#94A3B8"
$river = ConvertTo-OfficeRgb "#2B6CB0"
$storage = ConvertTo-OfficeRgb "#1D4ED8"
$dam = ConvertTo-OfficeRgb "#111827"
$niip = ConvertTo-OfficeRgb "#2CA25F"
$spr = ConvertTo-OfficeRgb "#7C3AED"
$hydro = ConvertTo-OfficeRgb "#D97706"
$esa = ConvertTo-OfficeRgb "#0891B2"
$flood = ConvertTo-OfficeRgb "#B91C1C"
$white = ConvertTo-OfficeRgb "#FFFFFF"
$lakeFill = ConvertTo-OfficeRgb "#DBEAFE"
$reservoirFill = ConvertTo-OfficeRgb "#BFDBFE"
$le = [char]0x2264
$ge = [char]0x2265

$powerPoint = $null
$presentation = $null
$slide = $null

try {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    $powerPoint.Visible = -1
    $presentation = $powerPoint.Presentations.Add()

    $slideWidth = 850.0
    $slideHeight = 580.0
    $presentation.PageSetup.SlideWidth = $slideWidth
    $presentation.PageSetup.SlideHeight = $slideHeight
    $slide = $presentation.Slides.Add(1, 12)

    # Main stem and downstream arrow.
    $mainstem = Add-EditableLine $slide "System - San Juan mainstem" 0.075 0.49 0.905 0.49 $slideWidth $slideHeight $river 3.2 $false
    $downstreamArrow = Add-EditableLine $slide "System - downstream arrow" 0.095 0.49 0.071 0.49 $slideWidth $slideHeight $river 2.2 $true

    # Animas tributary and NIIP diversion branches.
    Add-EditableLine $slide "System - Animas tributary" 0.585 0.415 0.585 0.49 $slideWidth $slideHeight $river 2.4 $false | Out-Null
    Add-EditableLine $slide "System - NIIP diversion branch" 0.905 0.49 0.925 0.415 $slideWidth $slideHeight $niip 2.4 $false | Out-Null

    # Lake Powell endpoint.
    $lakeNames = [System.Collections.Generic.List[string]]::new()
    $lake = $slide.Shapes.AddShape(5, 0.015 * $slideWidth, (1.0 - 0.456 - 0.068) * $slideHeight, 0.105 * $slideWidth, 0.068 * $slideHeight)
    $lake.Name = "Lake Powell - shape"
    $lake.Fill.Solid()
    $lake.Fill.ForeColor.RGB = $lakeFill
    $lake.Line.ForeColor.RGB = $river
    $lake.Line.Weight = 1.5
    $lakeNames.Add($lake.Name)
    $lakeNames.Add((Add-EditableText $slide "Lake Powell - label" "Lake Powell" 0.015 0.456 0.105 0.068 $slideWidth $slideHeight 13.0 $river $true 2).Name)
    Group-NamedShapes $slide $lakeNames.ToArray() "Lake Powell" | Out-Null

    # Navajo Reservoir endpoint.
    $reservoirNames = [System.Collections.Generic.List[string]]::new()
    $reservoir = $slide.Shapes.AddShape(5, 0.87 * $slideWidth, (1.0 - 0.452 - 0.076) * $slideHeight, 0.105 * $slideWidth, 0.076 * $slideHeight)
    $reservoir.Name = "Navajo Reservoir - shape"
    $reservoir.Fill.Solid()
    $reservoir.Fill.ForeColor.RGB = $reservoirFill
    $reservoir.Line.ForeColor.RGB = $storage
    $reservoir.Line.Weight = 1.6
    $reservoirNames.Add($reservoir.Name)
    $reservoirNames.Add((Add-EditableText $slide "Navajo Reservoir - label" "Navajo`nReservoir" 0.87 0.452 0.105 0.076 $slideWidth $slideHeight 12.5 $storage $true 2).Name)
    Group-NamedShapes $slide $reservoirNames.ToArray() "Navajo Reservoir" | Out-Null

    # One deliberately simple dam symbol.
    Add-EditableLine $slide "Navajo Dam - symbol" 0.835 0.449 0.835 0.531 $slideWidth $slideHeight $dam 4.0 $false | Out-Null
    Add-EditableText $slide "Navajo Dam - label" "Navajo Dam" 0.785 0.405 0.100 0.040 $slideWidth $slideHeight 10.5 $muted $false 2 | Out-Null

    # Gages and labels. Bluff and Farmington are emphasized as objective-evaluation locations.
    $gageSpecs = @(
        @{ Key = "Bluff"; X = 0.205; Label = "SJ near`nBluff"; Eval = $true },
        @{ Key = "Four Corners"; X = 0.335; Label = "SJ at`nFour Corners"; Eval = $false },
        @{ Key = "Shiprock"; X = 0.455; Label = "SJ at`nShiprock"; Eval = $false },
        @{ Key = "Farmington"; X = 0.585; Label = "SJ at`nFarmington"; Eval = $true },
        @{ Key = "Archuleta"; X = 0.705; Label = "SJ near`nArchuleta"; Eval = $false }
    )
    foreach ($gageSpec in $gageSpecs) {
        $size = if ($gageSpec.Eval) { 10.0 } else { 8.0 }
        $edge = if ($gageSpec.Eval) { $textColor } else { $context }
        $dot = $slide.Shapes.AddShape(
            9,
            $gageSpec.X * $slideWidth - $size / 2.0,
            (1.0 - 0.49) * $slideHeight - $size / 2.0,
            $size,
            $size
        )
        $dot.Name = "Gage - $($gageSpec.Key) marker"
        $dot.Fill.Solid()
        $dot.Fill.ForeColor.RGB = $white
        $dot.Line.ForeColor.RGB = $edge
        $dot.Line.Weight = if ($gageSpec.Eval) { 1.4 } else { 1.1 }
        $labelColor = if ($gageSpec.Eval) { $textColor } else { $muted }
        Add-EditableText $slide "Gage - $($gageSpec.Key) label" $gageSpec.Label ($gageSpec.X - 0.050) 0.401 0.100 0.070 $slideWidth $slideHeight 10.5 $labelColor $gageSpec.Eval 2 | Out-Null
    }

    $animasSize = 8.0
    $animasDot = $slide.Shapes.AddShape(9, 0.585 * $slideWidth - $animasSize / 2.0, (1.0 - 0.415) * $slideHeight - $animasSize / 2.0, $animasSize, $animasSize)
    $animasDot.Name = "Gage - Animas at Farmington marker"
    $animasDot.Fill.Solid()
    $animasDot.Fill.ForeColor.RGB = $white
    $animasDot.Line.ForeColor.RGB = $context
    $animasDot.Line.Weight = 1.1
    Add-EditableText $slide "Gage - Animas at Farmington label" "Animas at Farmington" 0.425 0.360 0.140 0.038 $slideWidth $slideHeight 10.0 $muted $false 3 | Out-Null
    Add-EditableText $slide "NIIP diversion - label" "NIIP diversion" 0.790 0.365 0.125 0.038 $slideWidth $slideHeight 10.0 $niip $true 3 | Out-Null
    Add-EditableText $slide "System - river label" "San Juan River" 0.430 0.515 0.140 0.035 $slideWidth $slideHeight 11.0 $river $true 2 | Out-Null
    Add-EditableText $slide "System - downstream label" "downstream" 0.130 0.510 0.100 0.030 $slideWidth $slideHeight 9.5 $muted $false 1 | Out-Null

    # Editable objective-to-location connector groups.
    Add-EditableConnector $slide "Connector - Flood safety to Bluff" @(@(0.187, 0.745), @(0.187, 0.725), @(0.205, 0.725), @(0.205, 0.49)) $slideWidth $slideHeight $flood
    Add-EditableConnector $slide "Connector - Flood safety to Farmington" @(@(0.187, 0.725), @(0.573, 0.725), @(0.573, 0.496)) $slideWidth $slideHeight $flood
    Add-EditableConnector $slide "Connector - ESA minimum flow to Farmington" @(@(0.468, 0.745), @(0.468, 0.700), @(0.597, 0.700), @(0.597, 0.496)) $slideWidth $slideHeight $esa
    Add-EditableConnector $slide "Connector - Storage to reservoir" @(@(0.698, 0.745), @(0.698, 0.680), @(0.885, 0.680), @(0.885, 0.516)) $slideWidth $slideHeight $storage
    Add-EditableConnector $slide "Connector - Dam safety to reservoir" @(@(0.925, 0.745), @(0.925, 0.520)) $slideWidth $slideHeight $dam
    Add-EditableConnector $slide "Connector - SPR to Farmington" @(@(0.492, 0.320), @(0.492, 0.350), @(0.585, 0.350), @(0.585, 0.484)) $slideWidth $slideHeight $spr
    Add-EditableConnector $slide "Connector - Hydropower to dam" @(@(0.710, 0.280), @(0.710, 0.335), @(0.817, 0.335), @(0.817, 0.480)) $slideWidth $slideHeight $hydro
    Add-EditableConnector $slide "Connector - NIIP delivery to diversion" @(@(0.900, 0.280), @(0.900, 0.345), @(0.925, 0.345), @(0.925, 0.415)) $slideWidth $slideHeight $niip

    # Objective cards are added after connectors so their white bodies mask any routed lines.
    Add-ObjectiveCard $slide "Objective - Flood safety" "Flood safety" @(
        "Keep discharge below both caps",
        "SJ at Farmington $le 5,000 cfs",
        "SJ near Bluff $le 12,000 cfs"
    ) "Priority 3" 0.045 0.745 0.285 0.175 $slideWidth $slideHeight $flood $textColor $muted $false 11.5 14.0

    Add-ObjectiveCard $slide "Objective - ESA minimum flow" "ESA minimum flow" @(
        "Maintain the daily minimum",
        "SJ at Farmington $ge 500 cfs"
    ) "Priority 2" 0.365 0.745 0.205 0.175 $slideWidth $slideHeight $esa $textColor $muted $false 11.5 14.0

    Add-ObjectiveCard $slide "Objective - Storage" "Storage" @(
        "Maintain carryover storage",
        "87.5% of maximum target",
        "98% soft upper guard"
    ) "Model-enabling" 0.605 0.745 0.185 0.175 $slideWidth $slideHeight $storage $textColor $muted $true 11.0 14.0

    Add-ObjectiveCard $slide "Objective - Dam safety" "Dam safety" @(
        "Avoid spill",
        "Spill = 0",
        "Navajo Reservoir"
    ) "Priority 1" 0.825 0.745 0.140 0.175 $slideWidth $slideHeight $dam $textColor $muted $false 11.0 13.0

    Add-ObjectiveCard $slide "Objective - Spring peak release" "Spring peak release" @(
        "Match annual event-attainment frequencies",
        "10,000 cfs / 5 d: 20%     8,000 cfs / 10 d: 33%",
        "5,000 cfs / 21 d: 50%     2,500 cfs / 10 d: 80%",
        "SJ at Farmington"
    ) "Priority 2" 0.235 0.075 0.355 0.245 $slideWidth $slideHeight $spr $textColor $muted $false 11.0 14.0

    Add-ObjectiveCard $slide "Objective - Hydropower" "Hydropower" @(
        "Maximize efficient generation",
        "Hydropower /",
        "maximum possible",
        "Navajo Dam"
    ) "Priority 3" 0.625 0.075 0.170 0.205 $slideWidth $slideHeight $hydro $textColor $muted $false 11.0 13.5

    Add-ObjectiveCard $slide "Objective - NIIP delivery" "NIIP delivery" @(
        "Meet annual agricultural",
        "delivery volume",
        "Annual volume / contract",
        "$ge 100%",
        "NIIP diversion"
    ) "Priority 1" 0.825 0.075 0.150 0.205 $slideWidth $slideHeight $niip $textColor $muted $false 10.5 12.5

    $presentation.SaveAs($output, 24)
}
finally {
    if ($presentation -ne $null) {
        try { $presentation.Close() } catch { }
    }
    if ($powerPoint -ne $null) {
        try { $powerPoint.Quit() } catch { }
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
