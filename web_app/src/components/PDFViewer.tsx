import { useState } from "react";
import { pdfjs, Document, Page } from "react-pdf";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import "react-pdf/dist/esm/Page/TextLayer.css";
import type { PDFDocumentProxy } from "pdfjs-dist";
import { ActionIcon, Box, Group, Loader, Skeleton, Text } from "@mantine/core";
import { IconChevronLeft, IconChevronRight } from "@tabler/icons-react";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.js",
  import.meta.url,
).toString();

const options = {
  cMapUrl: "/cmaps/",
  standardFontDataUrl: "/standard_fonts/",
};

type PDFFile = string | File | null;

export default function PDFViewer({ file }: { file: PDFFile }) {
  const [numPages, setNumPages] = useState<number>();
  const [pageNumber, setPageNumber] = useState(1);
  const [loading, setLoading] = useState(false);

  function onDocumentLoadSuccess({
    numPages: nextNumPages,
  }: PDFDocumentProxy): void {
    setNumPages(nextNumPages);
    setPageNumber(1);
    setLoading(false);
  }

  function changePage(offset: number) {
    setPageNumber((prevPageNumber) => prevPageNumber + offset);
  }

  function previousPage() {
    changePage(-1);
  }

  function nextPage() {
    changePage(1);
  }

  return (
    <Box>
      <Document
        file={file}
        loading={<Loader />}
        onLoadStart={() => setLoading(true)}
        onLoadSuccess={onDocumentLoadSuccess}
        options={options}
      >
        <Page loading={<Loader />} pageNumber={pageNumber} width={600} />
      </Document>
      <Group mt="md">
        <ActionIcon
          loading={loading}
          onClick={previousPage}
          disabled={pageNumber <= 1}
        >
          <IconChevronLeft />
        </ActionIcon>
        <Text>
          Page {pageNumber} of {numPages || "..."}
        </Text>
        <ActionIcon
          loading={loading}
          onClick={nextPage}
          disabled={!!numPages && pageNumber >= numPages}
        >
          <IconChevronRight />
        </ActionIcon>
      </Group>
    </Box>
  );
}
