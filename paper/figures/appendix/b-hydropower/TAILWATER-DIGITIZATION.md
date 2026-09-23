# Proposed tailwater rating points

`tailwater-knots.csv` records a fresh, approximate reading of Plate 7-4,
“Navajo Dam and Reservoir — Tailwater,” in the [2010 USACE Navajo water control
manual](https://water.usace.army.mil/cda/documents/wc/2560/Navajo_WCM_Draft_8-5-10Redacted.pdf).
The plate credits its curve to USBR Drawing 711-D-38. The working image was the
complete plate raster embedded on slide 2 of the historical hydropower-analysis
deck (513 × 777 pixels). That deck is optional provenance and is not needed to
rebuild the public figure package. These are proposed model knots, not numbers
tabulated by USACE or an exact tracing of the USBR drawing.

The raster's discharge axis runs from approximately x = 78.5 pixels at 0 cfs to x = 469 pixels at 40,000 cfs. Its elevation grid is approximately y = 456.5 pixels at 5,712 feet NGVD and y = 378 pixels at 5,714 feet NGVD, or about 39 pixels per foot. The CSV's pixel columns record approximate curve-center readings (image origin at upper left); the elevation column rounds them to 0.1 foot. The sharp low-flow bend is near 600 cfs and 5,713.0 feet. At 1,300 cfs the raster suggests about 5,713.2 feet, while interpolation between the proposed 600- and 2,000-cfs knots gives 5,713.15 feet, so a separate 1,300-cfs knot is unnecessary.

The bend occupies only about six horizontal pixels in this slide image. Treat its discharge position as uncertain by roughly 100–200 cfs and its elevation by roughly 0.1–0.2 foot; readings along the rest of the curve are uncertain by roughly 0.05–0.1 foot. The proposed polyline follows visual midpoints at 6,000–36,000 cfs within roughly 0.1 foot. The existing code's 2,000-, 4,000-, 32,000-, and 40,000-cfs elevations differ from this plate reading by about 0.8–1.1 feet. Refit the efficiency after replacing the coded rating points, then regenerate Appendix B figures and statistics from the updated model. Retain the original plate and this approximate-digitization limitation in the manuscript provenance.
